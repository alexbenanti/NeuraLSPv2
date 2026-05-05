import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_grid_side(A_csr: sp.csr_matrix) -> int:
    n = int(A_csr.shape[0])
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"Expected square structured grid, got {n} nodes.")
    return side


def _standardize_channels(channels: np.ndarray) -> np.ndarray:
    mean = channels.mean(axis=(0, 1), keepdims=True)
    std = channels.std(axis=(0, 1), keepdims=True)
    return (channels - mean) / (std + 1e-6)


def csr_to_grid_channels(A_csr: sp.csr_matrix, normalize: bool = True) -> np.ndarray:
    A_csr = sp.csr_matrix(A_csr)
    side = infer_grid_side(A_csr)
    n = side * side

    diag = np.zeros(n, dtype=np.float32)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    up = np.zeros(n, dtype=np.float32)
    down = np.zeros(n, dtype=np.float32)

    coo = A_csr.tocoo()
    for r, c, v in zip(coo.row, coo.col, coo.data):
        r = int(r)
        c = int(c)
        if r == c:
            diag[r] = np.float32(v)
        elif c == r - 1 and (r % side) != 0:
            left[r] = np.float32(v)
        elif c == r + 1 and (c % side) != 0:
            right[r] = np.float32(v)
        elif c == r - side:
            up[r] = np.float32(v)
        elif c == r + side:
            down[r] = np.float32(v)

    row_abs = np.asarray(np.abs(A_csr).sum(axis=1)).ravel().astype(np.float32)
    degree = np.diff(A_csr.indptr).astype(np.float32)

    channels = np.stack(
        [diag, left, right, up, down, row_abs, degree],
        axis=-1,
    ).reshape(side, side, 7)

    if normalize:
        channels = _standardize_channels(channels)
    return channels.astype(np.float32)


def grid_input_from_matrix(A_csr: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    x = csr_to_grid_channels(A_csr)
    return torch.from_numpy(x).unsqueeze(0).to(device)


@dataclass
class NeuralIFGraph:
    node_x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    diag_mask: torch.Tensor
    size: int


def build_neuralif_graph(A_csr: sp.csr_matrix, device: torch.device) -> NeuralIFGraph:
    A_csr = sp.csr_matrix(A_csr)
    side = infer_grid_side(A_csr)
    n = A_csr.shape[0]

    diag = A_csr.diagonal().astype(np.float32)
    degree = np.diff(A_csr.indptr).astype(np.float32)
    row_abs = np.asarray(np.abs(A_csr).sum(axis=1)).ravel().astype(np.float32)
    row_sum = np.asarray(A_csr.sum(axis=1)).ravel().astype(np.float32)

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, side, dtype=np.float32),
        np.linspace(0.0, 1.0, side, dtype=np.float32),
        indexing="ij",
    )

    node_x = np.stack(
        [
            diag,
            np.log1p(np.abs(diag)).astype(np.float32),
            degree / (degree.max() + 1e-6),
            row_abs / (row_abs.max() + 1e-6),
            row_sum / (np.max(np.abs(row_sum)) + 1e-6),
            xx.reshape(-1),
            yy.reshape(-1),
        ],
        axis=-1,
    )

    tril = sp.tril(A_csr, format="coo")
    edge_index = np.vstack([tril.row, tril.col]).astype(np.int64)
    edge_attr = tril.data.astype(np.float32)[:, None]
    diag_mask = edge_index[0] == edge_index[1]

    return NeuralIFGraph(
        node_x=torch.from_numpy(node_x).to(device),
        edge_index=torch.from_numpy(edge_index).long().to(device),
        edge_attr=torch.from_numpy(edge_attr).to(device),
        diag_mask=torch.from_numpy(diag_mask).bool().to(device),
        size=n,
    )


class ProjectionLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, Q: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        qt_s = torch.bmm(Q.transpose(1, 2), S)
        proj = torch.bmm(Q, qt_s)
        residual = S - proj
        per_sample = torch.linalg.vector_norm(residual.reshape(residual.shape[0], -1), dim=1)
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.fc1(x))
        y = self.fc2(y)
        return F.gelu(x + y)


class GreenfeldStencilMLP(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 100,
        depth: int = 12,
        output_rank: int = 72,
    ):
        super().__init__()
        self.output_rank = output_rank
        patch_dim = 9 * in_channels
        self.input_proj = nn.Linear(patch_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(depth)])
        self.output_proj = nn.Linear(hidden_dim, output_rank)

    def forward(self, x_grid: torch.Tensor) -> torch.Tensor:
        x = x_grid.permute(0, 3, 1, 2)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        patches = F.unfold(x, kernel_size=3).transpose(1, 2)

        y = F.gelu(self.input_proj(patches))
        for block in self.blocks:
            y = block(y)

        y = self.output_proj(y)
        q, _ = torch.linalg.qr(y, mode="reduced")
        return q


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights,
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class NeurKIttFNO(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        width: int = 48,
        modes1: int = 16,
        modes2: int = 16,
        num_layers: int = 4,
        output_rank: int = 72,
    ):
        super().__init__()
        self.output_rank = output_rank
        self.input_proj = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(num_layers)])
        self.output_proj = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, output_rank),
        )

    def forward(self, x_grid: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x_grid)
        x = x.permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 3, 1)
        y = self.output_proj(x).reshape(x.shape[0], -1, self.output_rank)
        q, _ = torch.linalg.qr(y, mode="reduced")
        return q


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2):
        super().__init__()
        modules = []
        d = in_dim
        for _ in range(layers - 1):
            modules.append(nn.Linear(d, hidden_dim))
            modules.append(nn.GELU())
            d = hidden_dim
        modules.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralIFMessagePassing(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_latent_dim: int,
        edge_skip_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.edge_latent_dim = edge_latent_dim
        self.edge_skip_dim = edge_skip_dim
        self.edge_update = MLP(
            2 * node_dim + edge_latent_dim + edge_skip_dim,
            hidden_dim,
            edge_latent_dim,
        )
        self.node_update = MLP(node_dim + edge_latent_dim, hidden_dim, node_dim)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_latent: torch.Tensor,
        edge_skip: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        edge_in = torch.cat([node_x[src], node_x[dst], edge_latent, edge_skip], dim=-1)
        e = self.edge_update(edge_in)
        agg = torch.zeros(
            node_x.shape[0], self.edge_latent_dim, device=node_x.device, dtype=node_x.dtype
        )
        agg.index_add_(0, dst, e)
        x = self.node_update(torch.cat([node_x, agg], dim=-1))
        return x, e


class NeuralIFFactorNet(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 7,
        edge_in_dim: int = 1,
        hidden_dim: int = 64,
        message_passing_steps: int = 3,
    ):
        super().__init__()
        self.node_encoder = MLP(node_in_dim, hidden_dim, hidden_dim)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
    [
        NeuralIFMessagePassing(
            node_dim=hidden_dim,
            edge_latent_dim=hidden_dim,
            edge_skip_dim=edge_in_dim,
            hidden_dim=hidden_dim,
        )
        for _ in range(message_passing_steps)
    ]
)
        self.edge_decoder = MLP(hidden_dim + edge_in_dim, hidden_dim, 1)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        diag_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_encoder(node_x)
        e0 = edge_attr
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x_new, e_new = layer(x, edge_index, e, e0)
            x = F.gelu(x + x_new)
            e = F.gelu(e + e_new)

        values = self.edge_decoder(torch.cat([e, e0], dim=-1)).squeeze(-1)
        diag_values = values[diag_mask]
        values = values.clone()
        values[diag_mask] = torch.exp(0.5 * diag_values)
        return values


def edge_values_to_torch_sparse(
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    size: int,
) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        edge_index,
        edge_values,
        size=(size, size),
        device=edge_values.device,
    ).coalesce()


def edge_values_to_csr(
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    size: int,
) -> sp.csr_matrix:
    rows = edge_index[0].detach().cpu().numpy()
    cols = edge_index[1].detach().cpu().numpy()
    vals = edge_values.detach().cpu().numpy().astype(np.float64)
    return sp.csr_matrix((vals, (rows, cols)), shape=(size, size))


def neuralif_sketched_loss(
    edge_index: torch.Tensor,
    edge_values: torch.Tensor,
    size: int,
    A_csr: sp.csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    L = edge_values_to_torch_sparse(edge_index, edge_values, size)
    Lt = torch.sparse_coo_tensor(
        torch.stack([edge_index[1], edge_index[0]], dim=0),
        edge_values,
        size=(size, size),
        device=device,
    ).coalesce()

    A_coo = A_csr.tocoo()
    A_t = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long().to(device),
        torch.from_numpy(A_coo.data.astype(np.float32)).to(device),
        size=(size, size),
        device=device,
    ).coalesce()

    z = torch.randn(size, 1, device=device)
    LLtz = torch.sparse.mm(L, torch.sparse.mm(Lt, z))
    Az = torch.sparse.mm(A_t, z)
    return torch.linalg.vector_norm(LLtz - Az)