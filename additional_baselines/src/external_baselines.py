import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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


def _real_orthonormal_basis_from_evecs(evecs: np.ndarray, target_rank: int) -> np.ndarray:
    blocks = []
    for j in range(evecs.shape[1]):
        col = np.asarray(evecs[:, j])
        if np.iscomplexobj(col):
            real = np.real(col).astype(np.float64, copy=False)
            imag = np.imag(col).astype(np.float64, copy=False)
            if np.linalg.norm(real) > 1e-12:
                blocks.append(real)
            if np.linalg.norm(imag) > 1e-12:
                blocks.append(imag)
        else:
            blocks.append(np.asarray(col, dtype=np.float64))

    if not blocks:
        raise ValueError('eigen-solver returned an empty basis')

    Y = np.column_stack(blocks)
    Q, _ = np.linalg.qr(Y, mode='reduced')
    if Q.shape[1] < target_rank:
        raise ValueError(
            f'only recovered {Q.shape[1]} real basis vectors for target rank {target_rank}'
        )
    return Q[:, :target_rank].astype(np.float32, copy=False)


def smallest_invariant_subspace_from_matrix(
    A_csr: sp.csr_matrix,
    target_rank: int,
    hermitian_tol: float = 1e-10,
    maxiter: Optional[int] = None,
) -> np.ndarray:
    """
    Build the NeurKItt target subspace directly from the matrix ``A``.
    The NeurKItt paper trains on the invariant subspace associated with the
    smallest eigenvalues of ``A`` (or smallest-magnitude eigenvalues in the
    non-Hermitian case).
    """
    A_csr = sp.csr_matrix(A_csr, dtype=np.float64)
    n = int(A_csr.shape[0])
    if n == 0:
        raise ValueError('cannot build an invariant subspace for an empty matrix')

    if n == 1:
        return np.ones((1, 1), dtype=np.float32)

    k = int(target_rank)
    if k <= 0:
        raise ValueError(f'target_rank must be positive, got {target_rank}')
    k = min(k, n - 1)

    asym = A_csr - A_csr.transpose()
    denom = float(np.linalg.norm(A_csr.data))
    if denom == 0.0:
        raise ValueError('cannot build an invariant subspace from a zero matrix')
    rel_asym = float(np.linalg.norm(asym.data) / denom) if asym.nnz else 0.0

    eig_kwargs = dict(k=k, which='SM', tol=1e-6)
    if maxiter is not None:
        eig_kwargs['maxiter'] = maxiter

    if rel_asym <= hermitian_tol:
        vals, vecs = spla.eigsh(A_csr, **eig_kwargs)
        order = np.argsort(np.abs(vals))
        basis = np.asarray(vecs[:, order], dtype=np.float64)
        Q, _ = np.linalg.qr(basis, mode='reduced')
        return Q[:, :k].astype(np.float32, copy=False)

    vals, vecs = spla.eigs(A_csr, **eig_kwargs)
    order = np.argsort(np.abs(vals))
    vecs = vecs[:, order]
    return _real_orthonormal_basis_from_evecs(vecs, k)


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


def csr_to_stencil_channels(A_csr: sp.csr_matrix, normalize: bool = False) -> np.ndarray:
    """
    Exact 3x3 stencil coefficients per node, arranged as
        [nw, n, ne, w, c, e, sw, s, se].
    This is the representation used by the Greenfeld paper.
    """
    A_csr = sp.csr_matrix(A_csr, dtype=np.float64)
    side = infer_grid_side(A_csr)
    n = side * side
    channels = np.zeros((n, 9), dtype=np.float32)

    rows = A_csr.indptr
    cols = A_csr.indices
    vals = A_csr.data
    offset_to_slot = {
        (-1, -1): 0,
        (-1, 0): 1,
        (-1, 1): 2,
        (0, -1): 3,
        (0, 0): 4,
        (0, 1): 5,
        (1, -1): 6,
        (1, 0): 7,
        (1, 1): 8,
    }

    for r in range(n):
        ri, rj = divmod(r, side)
        for ptr in range(rows[r], rows[r + 1]):
            c = int(cols[ptr])
            ci, cj = divmod(c, side)
            di = ci - ri
            dj = cj - rj
            slot = offset_to_slot.get((di, dj))
            if slot is not None:
                channels[r, slot] = np.float32(vals[ptr])

    channels = channels.reshape(side, side, 9)
    if normalize:
        channels = _standardize_channels(channels)
    return channels.astype(np.float32)


def grid_input_from_matrix(
    A_csr: sp.csr_matrix,
    device: torch.device,
    representation: str = "summary",
    normalize: bool = True,
) -> torch.Tensor:
    if representation == "summary":
        x = csr_to_grid_channels(A_csr, normalize=normalize)
    elif representation == "stencil":
        x = csr_to_stencil_channels(A_csr, normalize=normalize)
    else:
        raise ValueError(f"Unknown grid representation: {representation}")
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
        per_vector = torch.linalg.vector_norm(residual, dim=1)
        per_sample = per_vector.sum(dim=1)
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample.mean()


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        return F.relu(x + y)


def _safe_divide(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    sign = torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))
    den_safe = torch.where(den.abs() > eps, den, sign * eps)
    return num / den_safe


class GreenfeldStencilMLP(nn.Module):
    """
    Paper-aligned Greenfeld-style local stencil network.

    Input per coarse point:
      the 3x3 stencil of the coarse point itself and the stencils of its four
      immediate neighbors -> 5 * 9 = 45 scalars.

    Output per coarse point:
      four learned axis-neighbor interpolation weights. The four diagonal
      weights are then completed so that Au = 0 at those diagonal points,
      matching Section 3.1 of Greenfeld et al. 2019.
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim: int = 100,
        depth: int = 20,
        output_rank: Optional[int] = None,
        coarse_stride: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_rank = output_rank
        self.expected_rank = output_rank
        self.coarse_stride = coarse_stride

        self.input_proj = nn.Linear(5 * in_channels, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(depth)])
        self.output_proj = nn.Linear(hidden_dim, 4)  # [north, east, south, west]

    def _local_five_stencil_features(self, x_grid: torch.Tensor) -> torch.Tensor:
        x = x_grid.permute(0, 3, 1, 2)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        x = x.permute(0, 2, 3, 1)

        center = x[:, 1:-1, 1:-1, :]
        north = x[:, 0:-2, 1:-1, :]
        south = x[:, 2:, 1:-1, :]
        west = x[:, 1:-1, 0:-2, :]
        east = x[:, 1:-1, 2:, :]
        return torch.cat([center, north, south, west, east], dim=-1)

    @staticmethod
    def _pair_normalize(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        denom = a + b
        small = denom.abs() <= eps
        a_n = torch.where(small, 0.5 * torch.ones_like(a), a / denom)
        b_n = torch.where(small, 0.5 * torch.ones_like(b), b / denom)
        return a_n, b_n

    def _normalize_axis_weights(self, axis_weights: torch.Tensor) -> torch.Tensor:
        eff = axis_weights.clone()
        _, H, W, _ = axis_weights.shape
        coarse_rows = list(range(0, H, self.coarse_stride))
        coarse_cols = list(range(0, W, self.coarse_stride))

        for i in coarse_rows:
            for j_left, j_right in zip(coarse_cols[:-1], coarse_cols[1:]):
                left = axis_weights[:, i, j_left, 1]
                right = axis_weights[:, i, j_right, 3]
                left_n, right_n = self._pair_normalize(left, right)
                eff[:, i, j_left, 1] = left_n
                eff[:, i, j_right, 3] = right_n

        for i_up, i_down in zip(coarse_rows[:-1], coarse_rows[1:]):
            for j in coarse_cols:
                up = axis_weights[:, i_up, j, 2]
                down = axis_weights[:, i_down, j, 0]
                up_n, down_n = self._pair_normalize(up, down)
                eff[:, i_up, j, 2] = up_n
                eff[:, i_down, j, 0] = down_n

        return eff

    def _assemble_prolongation(self, axis_weights: torch.Tensor, stencils: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = axis_weights.shape
        coarse_rows = list(range(0, H, self.coarse_stride))
        coarse_cols = list(range(0, W, self.coarse_stride))
        nc = len(coarse_rows) * len(coarse_cols)

        if self.expected_rank is not None and nc != self.expected_rank:
            raise ValueError(
                f"Greenfeld expected coarse rank {self.expected_rank}, got {nc} for grid {H}x{W}."
            )

        axis_weights = self._normalize_axis_weights(axis_weights)
        P = axis_weights.new_zeros((B, H * W, nc))

        def flat(i: int, j: int) -> int:
            return i * W + j

        col = 0
        for i in coarse_rows:
            for j in coarse_cols:
                P[:, flat(i, j), col] = 1.0

                wn = axis_weights[:, i, j, 0]
                we = axis_weights[:, i, j, 1]
                ws = axis_weights[:, i, j, 2]
                ww = axis_weights[:, i, j, 3]

                if i - 1 >= 0:
                    P[:, flat(i - 1, j), col] = wn
                if j + 1 < W:
                    P[:, flat(i, j + 1), col] = we
                if i + 1 < H:
                    P[:, flat(i + 1, j), col] = ws
                if j - 1 >= 0:
                    P[:, flat(i, j - 1), col] = ww

                if i - 1 >= 0 and j - 1 >= 0:
                    s_nw = stencils[:, i - 1, j - 1, :]
                    d_nw = -_safe_divide(s_nw[:, 5] * wn + s_nw[:, 7] * ww + s_nw[:, 8], s_nw[:, 4])
                    P[:, flat(i - 1, j - 1), col] = d_nw

                if i - 1 >= 0 and j + 1 < W:
                    s_ne = stencils[:, i - 1, j + 1, :]
                    d_ne = -_safe_divide(s_ne[:, 3] * wn + s_ne[:, 7] * we + s_ne[:, 6], s_ne[:, 4])
                    P[:, flat(i - 1, j + 1), col] = d_ne

                if i + 1 < H and j - 1 >= 0:
                    s_sw = stencils[:, i + 1, j - 1, :]
                    d_sw = -_safe_divide(s_sw[:, 1] * ww + s_sw[:, 5] * ws + s_sw[:, 2], s_sw[:, 4])
                    P[:, flat(i + 1, j - 1), col] = d_sw

                if i + 1 < H and j + 1 < W:
                    s_se = stencils[:, i + 1, j + 1, :]
                    d_se = -_safe_divide(s_se[:, 1] * we + s_se[:, 3] * ws + s_se[:, 0], s_se[:, 4])
                    P[:, flat(i + 1, j + 1), col] = d_se

                col += 1

        return P

    def forward(self, x_grid: torch.Tensor) -> torch.Tensor:
        if x_grid.ndim != 4:
            raise ValueError(f"Expected x_grid with shape (B, H, W, C), got {tuple(x_grid.shape)}")
        if x_grid.shape[-1] != self.in_channels:
            raise ValueError(
                f"GreenfeldStencilMLP expected {self.in_channels} channels, got {x_grid.shape[-1]}"
            )

        feat = self._local_five_stencil_features(x_grid)
        B, H, W, Fdim = feat.shape
        y = feat.reshape(B * H * W, Fdim)
        y = F.relu(self.input_proj(y))
        for block in self.blocks:
            y = block(y)
        axis_weights = self.output_proj(y).reshape(B, H, W, 4)
        return self._assemble_prolongation(axis_weights, x_grid)


class GreenfeldFrobeniusLoss(nn.Module):
    """
    Paper-aligned Greenfeld objective:
      squared Frobenius norm of the two-grid error-propagation matrix
      M = S C S, where S is one Gauss-Seidel sweep and C is the Galerkin
      coarse-grid correction.
    """

    def __init__(self, jitter: float = 1e-8, normalize: bool = True):
        super().__init__()
        self.jitter = float(jitter)
        self.normalize = bool(normalize)

    def forward(self, P_batch: torch.Tensor, A_csr: sp.csr_matrix) -> torch.Tensor:
        if P_batch.ndim != 3:
            raise ValueError(f"Expected P with shape (B, n, nc), got {tuple(P_batch.shape)}")

        device = P_batch.device
        dtype = P_batch.dtype
        A = torch.as_tensor(A_csr.toarray(), device=device, dtype=dtype)
        n = A.shape[0]
        eye_n = torch.eye(n, device=device, dtype=dtype)

        L = torch.tril(A) + self.jitter * eye_n
        LA = torch.linalg.solve_triangular(L, A, upper=False)
        S = eye_n - LA

        losses = []
        for b in range(P_batch.shape[0]):
            P = P_batch[b]
            AP = A @ P
            PAP = P.transpose(0, 1) @ AP
            nc = PAP.shape[0]
            PAP = PAP + self.jitter * torch.eye(nc, device=device, dtype=dtype)
            coarse = torch.linalg.solve(PAP, P.transpose(0, 1) @ A)
            C = eye_n - P @ coarse
            M = S @ C @ S
            fro2 = torch.sum(M * M)
            if self.normalize:
                fro2 = fro2 / float(n * n)
            losses.append(fro2)

        return torch.stack(losses).mean()

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