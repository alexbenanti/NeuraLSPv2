import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import pyamg
import warnings

from src.pdes import generate_pde_data, smooth_test_vectors
from src.model import (
    ProlongationMLP2,
    nested_lora_loss,
    subspace_loss,
    error_propagation_loss,
)

from src.precorrector_baseline import (
    build_precorrector_ic0_graph,
    corrected_factor_to_csr,
)

from src.multigrid import TwoGridPreconditioner, pcg_solve

from src.external_baselines import (
    grid_input_from_matrix,
    build_neuralif_graph,
    edge_values_to_csr,
    smallest_invariant_subspace_from_matrix
)
from src.krylov_deflation import (
    solve_with_deflated_cg,
    solve_with_prebuilt_preconditioner,
)

# --- Optional GNN Import ---
try:
    from src.gnn_baseline import AMG_GNN
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("PyG / GNN Baseline not found. Skipping GNN experiments.")

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
N = 64
K_VECTORS = 72

# Train ONCE at this max rank, then test prefixes
RANKS = [2,4,6,8,10,12,14,20,24,26,28,30,32, 36,40,44,48,52,56,60,64,68,72]
#RANKS = [36,48,64,72]
RANK_MAX = max(RANKS)

TRAIN_EPOCHS = 1000
TEST_SAMPLES = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pde_type = "diffusion"
BENCH_MAXITER = 2000

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

SOR_OMEGA = 1.25
IC_SHIFT_INIT = 1e-10
IC_SHIFT_GROWTH = 10.0
IC_SHIFT_ATTEMPTS = 8

# If you want more stable timing on CPU:
# torch.set_num_threads(1)

# ==============================================================================
# FEATURES
# ==============================================================================
def get_features(A_csr, S_np):
    # Using only S as features (as in your current code)
    return torch.FloatTensor(S_np).to(DEVICE)

def _safe_triangular_solve(T_csr, rhs, *, lower):
    rhs = np.asarray(rhs, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        sol = spla.spsolve_triangular(T_csr, rhs, lower=lower)
    sol = np.asarray(sol, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(sol)):
        raise FloatingPointError("triangular solve returned non-finite values")
    return sol


def sanitize_lower_factor(L_csr, A_csr=None, rel_floor=1e-6, abs_floor=1e-10):
    L = sp.tril(sp.csr_matrix(L_csr, dtype=np.float64), format="csr").copy()
    L.data = np.nan_to_num(L.data, nan=0.0, posinf=0.0, neginf=0.0)

    d = np.abs(L.diagonal()).astype(np.float64)
    if A_csr is not None:
        A_diag = np.abs(sp.csr_matrix(A_csr, dtype=np.float64).diagonal()).astype(np.float64)
        nz = A_diag[A_diag > 0]
        scale = np.sqrt(np.median(nz)) if nz.size else 1.0
    else:
        nz = d[d > 0]
        scale = np.median(nz) if nz.size else 1.0

    floor = max(abs_floor, rel_floor * max(scale, 1.0))
    d = np.where(np.isfinite(d), d, 0.0)
    d = np.maximum(d, floor)
    L.setdiag(d)
    L.eliminate_zeros()
    return L


def _append_failure(stats, key):
    stats[key]["infer"].append(np.nan)
    stats[key]["setup"].append(np.nan)
    stats[key]["solve"].append(np.nan)
    stats[key]["total"].append(np.nan)
    stats[key]["iter"].append(np.nan)


def _safe_nanmedian(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmedian(x)) if np.any(np.isfinite(x)) else np.nan


def _safe_nanpercentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanpercentile(x, q)) if np.any(np.isfinite(x)) else np.nan

# ==============================================================================
# BASELINES
# ==============================================================================
def pyamg_inference(A, S):
    """
    PyAMG SA baseline. Returns PyAMG's prolongation P0 (dense float32).
    NOTE: PyAMG chooses coarse dimension automatically (not tied to RANKS).
    """
    B = np.ones((A.shape[0], 1), dtype=np.float64)
    ml = pyamg.smoothed_aggregation_solver(A, B=B, max_levels=2, max_coarse=10)
    P = ml.levels[0].P  # sparse

    if sp.issparse(P):
        P = P.toarray()
    P = np.asarray(P, dtype=np.float32)

    return P

def oracle_svd_Umax(S, rank_max):
    """
    Oracle coarse basis from top left singular vectors of S.
    U is already orthonormal.
    """
    U, _, _ = np.linalg.svd(S, full_matrices=False)
    return U[:, :rank_max].astype(np.float32)

def random_svd_U(S, rank, oversample=8, n_power_iter=1, seed=0, sketch_cap=None):
    """
    Randomized SVD approximation to the top `rank` left singular vectors of S.

    Notes
    -----
    * We use a private RNG seed so this baseline is reproducible without
      perturbing NumPy's global RNG state (which would change the benchmark).
    * The sweep rebuilds RandomSVD separately for each tested rank so the
      timed inference cost matches the current rank instead of a truncated
      larger factorization.
    * `sketch_cap` lets us prevent the randomized sketch from accidentally
      reaching the full column dimension, which would otherwise make a high-rank
      case collapse to the exact SVD before the full-rank point.
    """
    S = np.asarray(S, dtype=np.float64)
    m, n = S.shape
    max_rank = min(m, n)
    if not (1 <= rank <= max_rank):
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")

    if sketch_cap is None:
        sketch_dim = min(rank + oversample, max_rank)
    else:
        sketch_dim = min(rank + oversample, sketch_cap, max_rank)
        sketch_dim = max(rank, sketch_dim)

    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, sketch_dim))

    Y = S @ omega
    for _ in range(max(0, n_power_iter)):
        Y = S @ (S.T @ Y)

    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ S
    U_hat, _, _ = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat[:, :rank]
    return U.astype(np.float32)

def _safe_triangular_solve(T_csr, rhs, *, lower):
    rhs = np.asarray(rhs, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        sol = spla.spsolve_triangular(T_csr, rhs, lower=lower)
    sol = np.asarray(sol, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(sol)):
        raise FloatingPointError("triangular solve returned non-finite values")
    return sol


def sanitize_lower_factor(L_csr, A_csr=None, rel_floor=1e-6, abs_floor=1e-10):
    L = sp.tril(sp.csr_matrix(L_csr, dtype=np.float64), format="csr").copy()
    if not np.all(np.isfinite(L.data)):
        raise FloatingPointError("lower factor contains non-finite entries")

    d = np.abs(L.diagonal()).astype(np.float64)
    if A_csr is not None:
        A_diag = np.abs(sp.csr_matrix(A_csr, dtype=np.float64).diagonal()).astype(np.float64)
        nz = A_diag[A_diag > 0]
        scale = np.sqrt(np.median(nz)) if nz.size else 1.0
    else:
        nz = d[d > 0]
        scale = np.median(nz) if nz.size else 1.0

    floor = max(abs_floor, rel_floor * max(scale, 1.0))
    d = np.where(np.isfinite(d), d, 0.0)
    d = np.maximum(d, floor)
    L.setdiag(d)
    L.eliminate_zeros()
    return L


def _append_failure(stats, key):
    stats[key]["infer"].append(np.nan)
    stats[key]["setup"].append(np.nan)
    stats[key]["solve"].append(np.nan)
    stats[key]["total"].append(np.nan)
    stats[key]["iter"].append(np.nan)


def _safe_nanmedian(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmedian(x)) if np.any(np.isfinite(x)) else np.nan


def _safe_nanpercentile(x, q):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanpercentile(x, q)) if np.any(np.isfinite(x)) else np.nan

def _cg_solve_with_linear_preconditioner(A, b, apply_preconditioner, tol=1e-6, maxiter=None):
    """
    Small SciPy-based PCG helper used for the classical preconditioner baselines.
    """
    A_csr = sp.csr_matrix(A, dtype=np.float64)
    b_vec = np.asarray(b, dtype=np.float64)

    M = spla.LinearOperator(
        shape=A_csr.shape,
        matvec=lambda x: np.asarray(apply_preconditioner(np.asarray(x, dtype=np.float64)), dtype=np.float64),
        dtype=np.float64,
    )

    iters = [0]

    def callback(_xk):
        iters[0] += 1

    if maxiter is None:
        maxiter = 5 * A_csr.shape[0]

    t0 = time.perf_counter()
    try:
        _, info = spla.cg(A_csr, b_vec, M=M, rtol=tol, atol=0.0, maxiter=maxiter, callback=callback)
    except TypeError:
        # Older SciPy uses `tol` instead of `rtol`/`atol`.
        _, info = spla.cg(A_csr, b_vec, M=M, tol=tol, maxiter=maxiter, callback=callback)
    t_solve = time.perf_counter() - t0

    if info < 0:
        raise RuntimeError("CG failed with illegal input or breakdown.")
    if info > 0 and iters[0] == 0:
        iters[0] = info

    return t_solve, int(iters[0])


def _incomplete_cholesky0_factor(A, shift=0.0):
    """
    Simple sparse IC(0) factorization using the lower-triangular sparsity pattern of A.
    A positive diagonal shift can be supplied to avoid breakdown.
    """
    A_csr = sp.csr_matrix(A, dtype=np.float64)
    A_csr = 0.5 * (A_csr + A_csr.T)
    A_lower = sp.tril(A_csr, format="csr")
    n = A_lower.shape[0]

    strict_rows = [None] * n
    diag = np.zeros(n, dtype=np.float64)

    for i in range(n):
        start, end = A_lower.indptr[i], A_lower.indptr[i + 1]
        cols = A_lower.indices[start:end]
        vals = A_lower.data[start:end]

        a_ii = shift
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

    L = sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=A_csr.shape,
    )
    return L


def build_incomplete_cholesky_preconditioner(A):
    A_csr = sp.csr_matrix(A, dtype=np.float64)
    avg_diag = float(np.mean(np.abs(A_csr.diagonal())))
    shift = IC_SHIFT_INIT * max(1.0, avg_diag)
    last_err = None

    for _ in range(IC_SHIFT_ATTEMPTS):
        try:
            L_fact = _incomplete_cholesky0_factor(A_csr, shift=shift)
            L_fact = sanitize_lower_factor(L_fact, A_csr, rel_floor=1e-8, abs_floor=1e-12)
            Lt_fact = L_fact.transpose().tocsr()

            def apply_preconditioner(x, L=L_fact, Lt=Lt_fact):
                y = _safe_triangular_solve(L, x, lower=True)
                z = _safe_triangular_solve(Lt, y, lower=False)
                return np.asarray(z, dtype=np.float64)

            _ = apply_preconditioner(np.ones(A_csr.shape[0], dtype=np.float64))
            return apply_preconditioner
        except (np.linalg.LinAlgError, FloatingPointError, RuntimeWarning, ValueError) as err:
            last_err = err
            shift = max(1e-12, IC_SHIFT_GROWTH * max(shift, 1e-12))

    raise RuntimeError(
        f"Incomplete Cholesky failed after {IC_SHIFT_ATTEMPTS} shift attempts: {last_err}"
    )


def build_ssor_preconditioner(A, omega=SOR_OMEGA):
    """
    Build an SSOR preconditioner. The benchmark reports this as the SOR baseline,
    but uses the symmetric variant so it remains compatible with PCG.
    """
    if not (0.0 < omega < 2.0):
        raise ValueError(f"SOR/SSOR omega must lie in (0, 2), got {omega}")

    A_csr = sp.csr_matrix(A, dtype=np.float64)
    D = np.asarray(A_csr.diagonal(), dtype=np.float64)
    L = sp.tril(A_csr, k=-1, format="csr")
    U = sp.triu(A_csr, k=1, format="csr")

    DL = (L + sp.diags(D / omega)).tocsr()
    DU = (U + sp.diags(D / omega)).tocsr()
    scale = (2.0 - omega) / omega

    def apply_preconditioner(x):
        x = np.asarray(x, dtype=np.float64)
        y = _safe_triangular_solve(DL, x, lower=True)
        y = D * y
        z = _safe_triangular_solve(DU, y, lower=False)
        return scale * np.asarray(z, dtype=np.float64)

    return apply_preconditioner

# ==============================================================================
# TRAINING
# ==============================================================================
def train_model(model, optimizer, loss_fn, n_steps=50):
    t_start = time.perf_counter()
    model.train()
    loss_history = []

    print(f"  Training {model.__class__.__name__} for {n_steps} steps...")

    for step in range(n_steps):
        #A_csr, S_np = generate_pde_data(N, k_vectors=K_VECTORS, pde_type=pde_type)

        A_csr = generate_pde_data(N, pde_type=pde_type)

        S_np = smooth_test_vectors(A_csr, num_vectors = K_VECTORS)

        # Randomly permute columns of S so ordering doesn't leak
        perm = np.random.permutation(S_np.shape[1])
        S_np = S_np[:, perm]

        x = get_features(A_csr, S_np).unsqueeze(0)  # (1,n,K)
        S_target = torch.FloatTensor(S_np).unsqueeze(0).to(DEVICE)

        # Forward (handle GNN separately)
        if GNN_AVAILABLE and isinstance(model, torch.nn.Module) and "AMG_GNN" in str(type(model)):
            coo = A_csr.tocoo()
            edge_index = torch.tensor(
                np.vstack((coo.row, coo.col)), dtype=torch.long, device=DEVICE
            )
            edge_attr = torch.tensor(coo.data, dtype=torch.float, device=DEVICE).unsqueeze(1)

            x_flat = x.squeeze(0)  # (n,K)
            Q = model(x_flat, edge_index, edge_attr).unsqueeze(0)  # (1,n,r)

            
            if loss_fn.__name__ != "error_propagation_loss":
                Q, _ = torch.linalg.qr(Q)
        elif model.__class__.__name__ == "GreenfeldStencilMLP":
            x_grid = grid_input_from_matrix(A_csr, DEVICE, representation="stencil", normalize=False)
            Q = model(x_grid)
        elif model.__class__.__name__ == "NeurKIttFNO":
            x_grid = grid_input_from_matrix(A_csr, DEVICE, representation="summary", normalize=True)
            Q = model(x_grid)
            target_np = smallest_invariant_subspace_from_matrix(
                A_csr,
                target_rank=int(getattr(model, "output_rank")),
            )
            S_target = torch.from_numpy(target_np).unsqueeze(0).to(DEVICE)
        else:
            Q = model(x)  

        if loss_fn.__name__ == "error_propagation_loss" or loss_fn.__class__.__name__ == "GreenfeldFrobeniusLoss":
            loss = loss_fn(Q, A_csr)
        else:
            loss = loss_fn(Q, S_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

        if step == 0 or (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{n_steps} | Loss: {float(loss.item()):.6f}")

    return loss_history, time.perf_counter() - t_start

# ==============================================================================
# INFERENCE HELPERS (compute P_max ONCE, then slice)
# ==============================================================================
def infer_Pmax_mlp(model, A, S):
    model.eval()
    with torch.no_grad():
        x = get_features(A, S).unsqueeze(0)      # (1,n,K)
        Q = model(x).squeeze(0).cpu().numpy()    # (n,RANK_MAX), already orthonormal for ProlongationMLP2
    return Q.astype(np.float32)

def infer_Pmax_gnn(model, A, S, do_qr=False):
    """
    If do_qr=True, we QR the output to make prefix-truncation meaningful.
    If do_qr=False, matches your earlier baseline behavior when using error_propagation_loss.
    """
    model.eval()
    with torch.no_grad():
        x = get_features(A, S)  # (n,K)
        coo = A.tocoo()
        edge_index = torch.tensor(
            np.vstack((coo.row, coo.col)), dtype=torch.long, device=DEVICE
        )
        edge_attr = torch.tensor(coo.data, dtype=torch.float, device=DEVICE).unsqueeze(1)

        Y = model(x, edge_index, edge_attr)  # (n,RANK_MAX)
        if do_qr:
            Y, _ = torch.linalg.qr(Y)
        return Y.cpu().numpy().astype(np.float32)

def infer_grid_subspace(model, A, representation='summary', normalize=True):
    model.eval()
    with torch.no_grad():
        x_grid = grid_input_from_matrix(A, DEVICE, representation=representation, normalize=normalize)
        Q = model(x_grid).squeeze(0).cpu().numpy()
    return Q.astype(np.float32)


def infer_neuralif_factor(model, A):
    model.eval()
    with torch.no_grad():
        graph = build_neuralif_graph(A, DEVICE)
        edge_values = model(graph.node_x, graph.edge_index, graph.edge_attr, graph.diag_mask)
    return edge_values_to_csr(graph.edge_index, edge_values, graph.size)

def infer_precorrector_factor(model, A):
    model.eval()
    with torch.no_grad():
        graph = build_precorrector_ic0_graph(A, DEVICE)
        edge_values = model(graph.edge_index_msg, graph.edge_attr, graph.size)
    return corrected_factor_to_csr(graph.mat_index, edge_values, graph.size)


def solve_with_factorized_L(A, b, L_csr):
    t0 = time.perf_counter()
    L_csr = sanitize_lower_factor(L_csr, A, rel_floor=1e-3, abs_floor=1e-8)
    Lt = L_csr.transpose().tocsr()

    def apply_preconditioner(x):
        y = _safe_triangular_solve(L_csr, np.asarray(x, dtype=np.float64), lower=True)
        z = _safe_triangular_solve(Lt, np.asarray(y, dtype=np.float64), lower=False)
        return np.asarray(z, dtype=np.float64)

    _ = apply_preconditioner(np.ones(sp.csr_matrix(A).shape[0], dtype=np.float64))
    t_setup = time.perf_counter() - t0

    t_solve, iters = solve_with_prebuilt_preconditioner(
        A,
        b,
        apply_preconditioner,
        tol=1e-6,
        maxiter=min(5 * A.shape[0], BENCH_MAXITER),
    )
    return t_setup, t_solve, iters

# ==============================================================================
# SOLVE ONE INSTANCE WITH GIVEN P
# ==============================================================================
def solve_with_P(A, b, P):
    t0 = time.perf_counter()
    M = TwoGridPreconditioner(A, P)
    t_setup = time.perf_counter() - t0

    x_sol, hist, t_hist_solve = pcg_solve(A, b, M, tol=1e-6)
    t_solve = t_hist_solve[-1]
    iters = len(hist) - 1
    return t_setup, t_solve, iters

def solve_with_linear_preconditioner(A, b, builder_fn, maxiter_cap=BENCH_MAXITER):
    t0 = time.perf_counter()
    apply_preconditioner = builder_fn(A)
    t_setup = time.perf_counter() - t0

    t_solve, iters = solve_with_prebuilt_preconditioner(
        A,
        b,
        apply_preconditioner,
        tol=1e-6,
        maxiter=min(5 * A.shape[0], maxiter_cap),
    )
    return t_setup, t_solve, iters

# ==============================================================================
# RANK SWEEP BENCHMARK
# ==============================================================================

def run_rank_sweep(models, training_times, ranks, num_samples=100, pde_type=pde_type):
    print("\n========================================================")
    print(f"STARTING RANK-SWEEP BENCHMARK ({num_samples} samples)")
    print(f"PDE Type: {pde_type}")
    print(f"Train rank: {RANK_MAX}  |  Test ranks: {ranks}")
    print("========================================================")

    #auto_methods = {"PyAMG_SA", "Incomplete_Cholesky", "SOR"}

    auto_methods = {}

    method_keys = []
    for name, cfg in models.items():
        if cfg.get("rank_sweep", True):
            for r in ranks:
                method_keys.append(f"{name}_r={r}")
        else:
            method_keys.append(name)

    # for r in ranks:
    #     method_keys.append(f"Oracle_SVD_r={r}")
    #     method_keys.append(f"RandomSVD_r={r}")

    method_keys.extend(sorted(auto_methods))
    method_keys = list(dict.fromkeys(method_keys))

    stats = {k: {"infer": [], "setup": [], "solve": [], "total": [], "iter": []} for k in method_keys}
    pyamg_Pcols = []

    for i in range(num_samples):
        if (i+1)%10== 0 or i == 0:
            print(f"Processing sample {i+1}/{num_samples}...")

        A = generate_pde_data(N, pde_type=pde_type)
        b = np.random.randn(A.shape[0]).astype(np.float64)

        t0 = time.perf_counter()
        S = smooth_test_vectors(A, num_vectors=K_VECTORS)
        t_smooth_vectors = time.perf_counter() - t0

        t0 = time.perf_counter()
        Umax = oracle_svd_Umax(S, RANK_MAX)
        t_svd = time.perf_counter() - t0

        # try:
        #     t_set, t_sol, iters = solve_with_linear_preconditioner(A, b, build_incomplete_cholesky_preconditioner)
        #     stats["Incomplete_Cholesky"]["infer"].append(0.0)
        #     stats["Incomplete_Cholesky"]["setup"].append(t_set)
        #     stats["Incomplete_Cholesky"]["solve"].append(t_sol)
        #     stats["Incomplete_Cholesky"]["total"].append(t_set + t_sol)
        #     stats["Incomplete_Cholesky"]["iter"].append(iters)
        # except Exception as e:
        #     print(f"[warn] sample {i+1}: Incomplete_Cholesky failed: {e}")
        #     _append_failure(stats, "Incomplete_Cholesky")

        # try:
        #     t_set, t_sol, iters = solve_with_linear_preconditioner(
        #         A, b, lambda A_in: build_ssor_preconditioner(A_in, omega=SOR_OMEGA)
        #     )
        #     stats["SOR"]["infer"].append(0.0)
        #     stats["SOR"]["setup"].append(t_set)
        #     stats["SOR"]["solve"].append(t_sol)
        #     stats["SOR"]["total"].append(t_set + t_sol)
        #     stats["SOR"]["iter"].append(iters)
        # except Exception as e:
        #     print(f"[warn] sample {i+1}: SOR failed: {e}")
        #     _append_failure(stats, "SOR")

        # t0 = time.perf_counter()
        # P_pyamg = pyamg_inference(A, S)
        # t_pyamg = time.perf_counter() - t0
        # pyamg_Pcols.append(P_pyamg.shape[1])

        # t_set, t_sol, iters = solve_with_P(A, b, P_pyamg)
        # stats["PyAMG_SA"]["infer"].append(t_pyamg)
        # stats["PyAMG_SA"]["setup"].append(t_set)
        # stats["PyAMG_SA"]["solve"].append(t_sol)
        # stats["PyAMG_SA"]["total"].append(t_pyamg + t_set + t_sol)
        # stats["PyAMG_SA"]["iter"].append(iters)

        cache = {}
        for name, cfg in models.items():
            model = cfg["model"]
            kind = cfg.get("kind", "mlp")

            
            if kind == "gnn":
                t0 = time.perf_counter()
                artifact = infer_Pmax_gnn(model, A, S, do_qr=cfg.get("gnn_do_qr", False))
                t_inf = time.perf_counter() - t0
            elif kind == "mlp":
                t0 = time.perf_counter()
                artifact = infer_Pmax_mlp(model, A, S)
                t_inf = time.perf_counter() - t0
            elif kind in {"greenfeld", "neurkitt"}:
                t0 = time.perf_counter()
                artifact = infer_grid_subspace(
                    model,
                    A,
                    representation=cfg.get("input_representation", "summary"),
                    normalize=cfg.get("normalize_input", True),
                )
                t_inf = time.perf_counter() - t0
            elif kind == "neuralif":
                t0 = time.perf_counter()
                artifact = infer_neuralif_factor(model, A)
                t_inf = time.perf_counter() - t0

            elif kind == "precorrector":
                artifact = infer_precorrector_factor(model, A)
            else:
                raise ValueError(f"Unknown model kind: {kind}")
            

            cache[name] = (artifact, t_inf)

        for r in ranks:
            P = Umax[:, :r]
            # t_set, t_sol, iters = solve_with_P(A, b, P)
            # key = f"Oracle_SVD_r={r}"
            # stats[key]["infer"].append(t_svd)
            # stats[key]["setup"].append(t_set)
            # stats[key]["solve"].append(t_sol)
            # stats[key]["total"].append(t_svd + t_set + t_sol + t_smooth_vectors)
            # stats[key]["iter"].append(iters)

            t0 = time.perf_counter()
            sketch_cap = None if r >= S.shape[1] else max(1, S.shape[1] - 1)
            P = random_svd_U(
                S,
                r,
                oversample=8,
                n_power_iter=1,
                seed=SEED + 10_000 * i + r,
                sketch_cap=sketch_cap,
            )
            t_rsvd = time.perf_counter() - t0

            # t_set, t_sol, iters = solve_with_P(A, b, P)
            # key = f"RandomSVD_r={r}"
            # stats[key]["infer"].append(t_rsvd)
            # stats[key]["setup"].append(t_set)
            # stats[key]["solve"].append(t_sol)
            # stats[key]["total"].append(t_rsvd + t_set + t_sol + t_smooth_vectors)
            # stats[key]["iter"].append(iters)

            for name, cfg in models.items():
                if not cfg.get("rank_sweep", True):
                    continue

                artifact, t_inf = cache[name]
                basis = artifact[:, :r]

                if cfg.get("solver") == "deflated_cg":
                    t_set, t_sol, iters = solve_with_deflated_cg(
                        A, b, basis, tol=1e-6, maxiter=BENCH_MAXITER
                    )
                else:
                    t_set, t_sol, iters = solve_with_P(A, b, basis)

                total = t_inf + t_set + t_sol
                if cfg.get("uses_smooth_vectors", False):
                    total += t_smooth_vectors

                key = f"{name}_r={r}"
                stats[key]["infer"].append(t_inf)
                stats[key]["setup"].append(t_set)
                stats[key]["solve"].append(t_sol)
                stats[key]["total"].append(total)
                stats[key]["iter"].append(iters)

            for name, cfg in models.items():
                if cfg.get("rank_sweep", True):
                    continue

                artifact, t_inf = cache[name]
                solver = cfg.get("solver", "two_grid")

                if solver == "deflated_cg":
                    t_set, t_sol, iters = solve_with_deflated_cg(
                        A, b, artifact, tol=1e-6, maxiter=BENCH_MAXITER
                    )
                elif solver == "factorized":
                    t_set, t_sol, iters = solve_with_factorized_L(A, b, artifact)
                elif solver == "two_grid":
                    t_set, t_sol, iters = solve_with_P(A, b, artifact)
                else:
                    raise ValueError(f"Unknown fixed-size solver for {name}: {solver}")

                total = t_inf + t_set + t_sol
                if cfg.get("uses_smooth_vectors", False):
                    total += t_smooth_vectors

                stats[name]["infer"].append(t_inf)
                stats[name]["setup"].append(t_set)
                stats[name]["solve"].append(t_sol)
                stats[name]["total"].append(total)
                stats[name]["iter"].append(iters)

    rows = []
    for name, m in stats.items():
        if "_r=" in name:
            base, r_str = name.split("_r=")
            rank_val = int(r_str)
            train_time = training_times.get(base, 0.0)
        else:
            base = name
            if base in auto_methods:
                rank_val = "auto"
            else:
                rank_val = models[base].get("paper_rank", "paper")
            train_time = training_times.get(base, 0.0)

        rows.append({
            "Method": base,
            "Rank": rank_val,
            "Train Time (s)": train_time,
            "Inference (ms)": 1000 * _safe_nanmedian(m["infer"]),
            "Setup (ms)": 1000 * _safe_nanmedian(m["setup"]),
            "Solve (ms)": 1000 * _safe_nanmedian(m["solve"]),
            "Total (ms)": 1000 * _safe_nanmedian(m["total"]),
            "25th Percentile (ms)": 1000 * _safe_nanpercentile(m["total"], 25),
            "75th Percentile (ms)": 1000 * _safe_nanpercentile(m["total"], 75),
            "Median Iterations": _safe_nanmedian(m["iter"]),
        })

    df = pd.DataFrame(rows)

    # print(
    #     f"\n[PyAMG] P0 columns: median={int(np.median(pyamg_Pcols))}, "
    #     f"mean={np.mean(pyamg_Pcols):.1f}, min={min(pyamg_Pcols)}, max={max(pyamg_Pcols)}"
    # )
    return df
 

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    raise SystemExit("Use train_models.py followed by main.py after the baseline integration patch.")
