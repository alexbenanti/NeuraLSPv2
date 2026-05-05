from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2):
        super().__init__()
        modules = []
        d = in_dim
        for _ in range(max(0, layers - 1)):
            modules.append(nn.Linear(d, hidden_dim))
            modules.append(nn.Tanh())
            d = hidden_dim
        modules.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PreCorrectorGraph:
    edge_index_msg: torch.Tensor  # [2, E] with src=col, dst=row
    mat_index: torch.Tensor       # [2, E] with row, col
    edge_attr: torch.Tensor       # [E, 1], raw IC(0) entries
    size: int
    L0_csr: sp.csr_matrix


def _coerce_spd_matrix(A) -> sp.csr_matrix:
    A_csr = sp.csr_matrix(A, dtype=np.float64)
    return 0.5 * (A_csr + A_csr.T)


def _incomplete_cholesky0_factor(
    A,
    shift: float = 0.0,
) -> sp.csr_matrix:
    """
    Simple IC(0) factorization using the lower-triangular sparsity pattern of A.
    A small positive diagonal shift can be supplied to avoid numerical breakdown.
    """
    A_csr = _coerce_spd_matrix(A)
    A_lower = sp.tril(A_csr, format="csr")
    n = A_lower.shape[0]

    strict_rows = [None] * n
    diag = np.zeros(n, dtype=np.float64)

    for i in range(n):
        start, end = A_lower.indptr[i], A_lower.indptr[i + 1]
        cols = A_lower.indices[start:end]
        vals = A_lower.data[start:end]

        a_ii = float(shift)
        a_lower_row = {}
        for c, v in zip(cols, vals):
            c = int(c)
            if c < i:
                a_lower_row[c] = float(v)
            elif c == i:
                a_ii += float(v)

        Li = {}
        for j in sorted(a_lower_row.keys()):
            s = a_lower_row[j]
            Lj = strict_rows[j]
            if Li and Lj:
                if len(Li) <= len(Lj):
                    acc = 0.0
                    for k, lik in Li.items():
                        ljk = Lj.get(k)
                        if ljk is not None:
                            acc += lik * ljk
                else:
                    acc = 0.0
                    for k, ljk in Lj.items():
                        lik = Li.get(k)
                        if lik is not None:
                            acc += lik * ljk
                s -= acc

            lij = s / diag[j]
            if not np.isfinite(lij):
                raise np.linalg.LinAlgError("IC(0) breakdown: non-finite subdiagonal entry.")
            Li[j] = lij

        d = a_ii - sum(v * v for v in Li.values())
        if (not np.isfinite(d)) or d <= 0.0:
            raise np.linalg.LinAlgError("IC(0) breakdown: non-positive pivot.")

        diag[i] = np.sqrt(d)
        strict_rows[i] = Li

    data = []
    indices = []
    indptr = [0]
    for i in range(n):
        Li = strict_rows[i]
        if Li:
            for j in sorted(Li.keys()):
                indices.append(j)
                data.append(Li[j])
        indices.append(i)
        data.append(diag[i])
        indptr.append(len(indices))

    return sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=A_csr.shape,
    )


def build_ic0_factor(
    A,
    shift_init: float = 1e-10,
    shift_growth: float = 10.0,
    shift_attempts: int = 8,
) -> sp.csr_matrix:
    A_csr = _coerce_spd_matrix(A)
    avg_diag = float(np.mean(np.abs(A_csr.diagonal())))
    shift = shift_init * max(1.0, avg_diag)
    last_err = None

    for _ in range(shift_attempts):
        try:
            return _incomplete_cholesky0_factor(A_csr, shift=shift)
        except np.linalg.LinAlgError as err:
            last_err = err
            shift = max(1e-12, shift_growth * max(shift, 1e-12))

    raise RuntimeError(
        f"IC(0) failed after {shift_attempts} shift attempts: {last_err}"
    )


def build_precorrector_ic0_graph(A, device: torch.device) -> PreCorrectorGraph:
    L0_csr = build_ic0_factor(A)
    coo = L0_csr.tocoo()

    mat_index = np.vstack([coo.row, coo.col]).astype(np.int64)
    edge_index_msg = np.vstack([coo.col, coo.row]).astype(np.int64)
    edge_attr = coo.data.astype(np.float32)[:, None]

    return PreCorrectorGraph(
        edge_index_msg=torch.from_numpy(edge_index_msg).long().to(device),
        mat_index=torch.from_numpy(mat_index).long().to(device),
        edge_attr=torch.from_numpy(edge_attr).to(device),
        size=L0_csr.shape[0],
        L0_csr=L0_csr,
    )


def _scatter_max_1d(messages: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    messages: [E, F]
    dst: [E]
    returns [N, F]
    """
    feat_dim = messages.shape[1]

    if hasattr(torch.Tensor, "scatter_reduce_"):
        out = torch.full(
            (num_nodes, feat_dim),
            -torch.inf,
            dtype=messages.dtype,
            device=messages.device,
        )
        index = dst.view(-1, 1).expand(-1, feat_dim)
        out.scatter_reduce_(0, index, messages, reduce="amax", include_self=True)

        # IMPORTANT: out-of-place replacement, not in-place mutation
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out

    # Safe fallback for older PyTorch versions
    rows = []
    zero = torch.zeros((feat_dim,), dtype=messages.dtype, device=messages.device)
    for i in range(num_nodes):
        mask = (dst == i)
        if torch.any(mask):
            rows.append(messages[mask].max(dim=0).values)
        else:
            rows.append(zero)
    return torch.stack(rows, dim=0)


class PreCorrectorIC0(nn.Module):
    """
    Paper-inspired PreCorrector variant for IC(0) factors.

    The paper uses L(theta) = L + alpha * GNN(L), trains alpha from 0,
    uses ones as node inputs, and applies shared message passing rounds
    over the lower-triangular factor graph.
    """

    def __init__(self, hidden_dim: int = 16, num_rounds: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rounds = num_rounds

        # Parameter count matches the paper's compact PreCorrector design:
        # encoder 1->16->16, processor (16 + 1 + 1)->16->16, decoder 16->16->1, plus alpha.
        self.edge_encoder = MLP(1, hidden_dim, hidden_dim, layers=2)
        self.edge_update = MLP(hidden_dim + 2, hidden_dim, hidden_dim, layers=2)
        self.edge_decoder = MLP(hidden_dim, hidden_dim, 1, layers=2)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        edge_index_msg: torch.Tensor,
        edge_attr_raw: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        src, dst = edge_index_msg

        scale = torch.max(torch.abs(edge_attr_raw)).clamp_min(1e-8)
        e = self.edge_encoder(edge_attr_raw / scale)
        h = torch.ones((num_nodes, 1), dtype=edge_attr_raw.dtype, device=edge_attr_raw.device)

        for _ in range(self.num_rounds):
            edge_gate = torch.mean(e, dim=1, keepdim=True)
            messages = h[src] * edge_gate
            h = _scatter_max_1d(messages, dst, num_nodes)
            e = self.edge_update(torch.cat([e, h[dst], h[src]], dim=1))

        correction = scale * self.edge_decoder(e)
        corrected = edge_attr_raw + self.alpha * correction
        return corrected.squeeze(-1)


def edge_values_to_torch_sparse(
    mat_index: torch.Tensor,
    edge_values: torch.Tensor,
    size: int,
) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        mat_index,
        edge_values,
        size=(size, size),
        device=edge_values.device,
    ).coalesce()


def corrected_factor_to_csr(
    mat_index: torch.Tensor,
    edge_values: torch.Tensor,
    size: int,
) -> sp.csr_matrix:
    rows = mat_index[0].detach().cpu().numpy()
    cols = mat_index[1].detach().cpu().numpy()
    vals = edge_values.detach().cpu().numpy().astype(np.float64)
    return sp.csr_matrix((vals, (rows, cols)), shape=(size, size))


def precorrector_ic0_training_loss(
    model: PreCorrectorIC0,
    A_csr,
    device: torch.device,
    num_hutchinson_samples: int = 1,
) -> torch.Tensor:
    """
    Paper-faithful PreCorrector training loss.

    Target objective (Eq. 3):
        ||(P - A) A^{-1}||_F^2,  where P = L(theta)L(theta)^T

    Practical stochastic implementation (Eqs. 4-5):
        E_b ||P x - b||_2^2,  where b ~ N(0, I) and A x = b.

    We use the sampled form because the paper explicitly derives (5)
    from (3) to avoid explicit inverse materialization during training.
    """
    if num_hutchinson_samples < 1:
        raise ValueError("num_hutchinson_samples must be >= 1")

    A_csr = _coerce_spd_matrix(A_csr)
    graph = build_precorrector_ic0_graph(A_csr, device)
    edge_values = model(graph.edge_index_msg, graph.edge_attr, graph.size)

    L = edge_values_to_torch_sparse(graph.mat_index, edge_values, graph.size)
    Lt = edge_values_to_torch_sparse(
        torch.stack([graph.mat_index[1], graph.mat_index[0]], dim=0),
        edge_values,
        graph.size,
    )

    # Reuse one sparse factorization of A for all Hutchinson probes in this step.
    solve_A = spla.factorized(A_csr.tocsc())

    losses = []
    for _ in range(num_hutchinson_samples):
        b_np = np.random.randn(graph.size).astype(np.float64)
        x_np = np.asarray(solve_A(b_np), dtype=np.float64)

        b = torch.from_numpy(b_np.astype(np.float32)).to(device).unsqueeze(1)
        x = torch.from_numpy(x_np.astype(np.float32)).to(device).unsqueeze(1)

        Px = torch.sparse.mm(L, torch.sparse.mm(Lt, x))
        residual = Px - b
        losses.append(torch.sum(residual * residual))

    return torch.stack(losses).mean()