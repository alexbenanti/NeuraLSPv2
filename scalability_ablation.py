import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import pyamg
import re
import os
from datetime import datetime

from src.pdes import generate_pde_data, smooth_test_vectors
from src.model import (
    ProlongationMLP2,
    nested_lora_loss,
    subspace_loss,
    error_propagation_loss,
)
from src.multigrid import TwoGridPreconditioner, pcg_solve

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
TRAIN_EPOCHS = 1000
TEST_SAMPLES = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pde_type = "diffusion"

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

TOL = 1e-6   # same tolerance you use in pcg_solve in main.py

# --- Sizes you want to test ---
Ns = [16, 32, 48, 64, 80]


# ==============================================================================
# PER-N RULES  
# ==============================================================================
def K_for_N(N: int) -> int:
    # number of smoothed vectors
    return int(N)

def r_for_N(N: int) -> int:
    # target rank / coarse basis size
    return max(1, int(N // 2))

# ==============================================================================
# FEATURES
# ==============================================================================
def get_features(A_csr, S_np):
    # Using only S as features
    return torch.FloatTensor(S_np).to(DEVICE)

# ==============================================================================
# PYAMG BASELINE 
# ==============================================================================
def pyamg_inference(A, S):
    """
    PyAMG SA baseline. Returns PyAMG's prolongation P0 (dense float32).
    PyAMG chooses coarse dimension automatically (not tied to r_for_N).
    """
    A = A.tocsr()
    n = A.shape[0]

    B = np.ones((n, 1), dtype=np.float64)

    # Make sure max_coarse < n to avoid "single level" for small n
    max_coarse = min(10, n - 1)
    max_coarse = max(2, max_coarse)

    ml = pyamg.smoothed_aggregation_solver(A, B=B, max_levels=2, max_coarse=max_coarse)

    P = getattr(ml.levels[0], "P", None)
    if P is None:
        # Fallback (rare for your N>=16), but prevents crashes
        return np.eye(n, min(n, 2), dtype=np.float32)

    if sp.issparse(P):
        P = P.toarray()
    return np.asarray(P, dtype=np.float32)

# ==============================================================================
# TRAINING
# ==============================================================================
def train_model(model, N, K_vectors, optimizer, loss_fn, n_steps=50):
    """
    Trains a model on random PDE instances generated on the fly.
    Uses K_vectors = K_for_N(N).
    """
    t_start = time.perf_counter()
    model.train()
    loss_history = []

    print(f"  Training {model.__class__.__name__} for {n_steps} steps...")

    for _ in range(n_steps):
        A_csr = generate_pde_data(N, pde_type=pde_type)
        if isinstance(A_csr, tuple) and len(A_csr) == 2:
            A_csr, _ = A_csr

        S_np = smooth_test_vectors(A_csr, num_vectors=K_vectors)

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
        else:
            Q = model(x)  # ProlongationMLP2 returns QR’d Q already

        if loss_fn.__name__ == "error_propagation_loss":
            loss = loss_fn(Q, A_csr)
        else:
            loss = loss_fn(Q, S_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return loss_history, time.perf_counter() - t_start

# ==============================================================================
# EVALUATION HELPERS
# ==============================================================================
def iters_and_time_to_tol(hist, t_hist, tol=TOL):
    hist = np.asarray(hist, dtype=np.float64)
    t_hist = np.asarray(t_hist, dtype=np.float64)

    if len(hist) == 0:
        return 0, 0.0, False

    r0 = hist[0] if hist[0] != 0 else 1.0
    thresh = tol * r0

    for k in range(len(hist)):
        if hist[k] <= thresh:
            return k, float(t_hist[k]), True

    return len(hist) - 1, float(t_hist[-1]), False


def infer_mlp(model, A_csr, S_np):
    model.eval()
    with torch.no_grad():
        x = get_features(A_csr, S_np).unsqueeze(0)
        t0 = time.perf_counter()
        Q = model(x)
        t_inf = time.perf_counter() - t0
    P = Q.squeeze(0).cpu().numpy().astype(np.float32)
    return P, t_inf


def oracle_svd(A_csr, S_np, rank):
    t0 = time.perf_counter()
    U, _, _ = np.linalg.svd(S_np, full_matrices=False)
    P = U[:, :rank].astype(np.float32)
    t_inf = time.perf_counter() - t0
    return P, t_inf


def solve_with_P(A_csr, b, P, tol=TOL):
    t0 = time.perf_counter()
    M = TwoGridPreconditioner(A_csr, P)
    t_setup = time.perf_counter() - t0

    x_sol, hist, t_hist = pcg_solve(A_csr, b, M, tol=tol)
    iters_to_tol, t_solve_to_tol, conv = iters_and_time_to_tol(hist, t_hist, tol=tol)
    return t_setup, t_solve_to_tol, iters_to_tol, conv


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # -----------------------
    # Define models per N (K=N, r=N/2)   ### CHANGED ###
    # -----------------------
    models = {}
    for N in Ns:
        K_vec = K_for_N(N)
        r_tgt = r_for_N(N)
        n_nodes = (N + 1) ** 2

        # NOTE: This global-MLP architecture scales VERY badly with n_nodes*K_vec.
        # If you hit memory limits at large N, reduce hidden dims or reduce Ns.

        mlp_nested = ProlongationMLP2(K_vec, 128, 256, r_tgt, n_nodes, r_tgt).to(DEVICE)
        name = f"NLSS_N={N}"
        models[name] = {
            "model": mlp_nested,
            "loss": nested_lora_loss,
            "opt": optim.Adam(mlp_nested.parameters(), lr=LR),
            "kind": "mlp",
            "N": N,
            "K": K_vec,
            "r": r_tgt,
        }

    # -----------------------
    # Train
    # -----------------------
    training_times = {}
    print(f"Starting Training on {DEVICE}...")
    for name, cfg in models.items():
        print(f"\n== Training {name}: N={cfg['N']} | K={cfg['K']} | r={cfg['r']} ==")
        _, t_train = train_model(
            cfg["model"],
            N=cfg["N"],
            K_vectors=cfg["K"],
            optimizer=cfg["opt"],
            loss_fn=cfg["loss"],
            n_steps=TRAIN_EPOCHS,
        )
        training_times[name] = t_train
        print(f"  -> {name} train time: {t_train:.2f}s")

    # -----------------------
    # Evaluation stats  ### CHANGED: record coarse_dim ###
    # -----------------------
    USE_ORACLE_SVD = True
    USE_PYAMG = True

    stats = {}

    def init_stats_key(N, method):
        stats[(N, method)] = {
            "t_A": [],
            "t_S": [],
            "t_inf": [],
            "t_setup": [],
            "t_solve_tol": [],
            "t_pipeline": [],
            "t_e2e": [],
            "iters_tol": [],
            "converged": [],
            "coarse_dim": [],   # <-- NEW
        }

    def push(stats_key, t_A, t_S, t_inf, t_setup, t_solve, iters, conv, coarse_dim, include_S_in_e2e=True):
        d = stats[stats_key]
        d["t_A"].append(t_A)
        d["t_S"].append(t_S)
        d["t_inf"].append(t_inf)
        d["t_setup"].append(t_setup)
        d["t_solve_tol"].append(t_solve)
        d["t_pipeline"].append(t_inf + t_setup + t_solve)
        if include_S_in_e2e:
            d["t_e2e"].append(t_A + t_S + t_inf + t_setup + t_solve)
        else:
            d["t_e2e"].append(t_A + t_inf + t_setup + t_solve)
        d["iters_tol"].append(iters)
        d["converged"].append(1.0 if conv else 0.0)
        d["coarse_dim"].append(int(coarse_dim))

    print("\n" + "=" * 70)
    print("STARTING SCALABILITY EVALUATION")
    print(f"PDE={pde_type} | Rule: K=N, r=N/2 | Tol={TOL} | Samples={TEST_SAMPLES}")
    print("=" * 70)

    Ns_unique = sorted(set(cfg["N"] for cfg in models.values()))

    for N in Ns_unique:
        K_vec = K_for_N(N)
        r_tgt = r_for_N(N)
        n = (N + 1) ** 2

        print(f"\n--- Evaluating N={N} (n={n}) | K={K_vec} | r={r_tgt} ---")

        model_names = [name for name, cfg in models.items() if cfg["N"] == N]
        for name in model_names:
            init_stats_key(N, name)
        if USE_ORACLE_SVD:
            init_stats_key(N, "SVD")
        if USE_PYAMG:
            init_stats_key(N, "PyAMG_SA")

        for i in range(TEST_SAMPLES):
            if i % 10 == 0:
                print(f"  sample {i}/{TEST_SAMPLES}")

            # ---- Generate A ----
            t0 = time.perf_counter()
            A_csr = generate_pde_data(N, pde_type=pde_type)
            if isinstance(A_csr, tuple) and len(A_csr) == 2:
                A_csr, _ = A_csr
            t_A = time.perf_counter() - t0

            # ---- Generate S with K=N ----
            t0 = time.perf_counter()
            S_np = smooth_test_vectors(A_csr, num_vectors=K_vec)
            t_S = time.perf_counter() - t0

            b = np.random.randn(A_csr.shape[0]).astype(np.float32)

            # ---- Learned model(s) for this N ----
            for name in model_names:
                model = models[name]["model"]
                P, t_inf = infer_mlp(model, A_csr, S_np)
                t_setup, t_solve, iters, conv = solve_with_P(A_csr, b, P, tol=TOL)

                push(
                    (N, name),
                    t_A=t_A, t_S=t_S, t_inf=t_inf,
                    t_setup=t_setup, t_solve=t_solve,
                    iters=iters, conv=conv,
                    coarse_dim=P.shape[1],
                    include_S_in_e2e=True,
                )

            # ---- Oracle SVD baseline (rank = r=N/2) ----
            if USE_ORACLE_SVD:
                P, t_inf = oracle_svd(A_csr, S_np, rank=r_tgt)
                t_setup, t_solve, iters, conv = solve_with_P(A_csr, b, P, tol=TOL)

                push(
                    (N, "SVD"),
                    t_A=t_A, t_S=t_S, t_inf=t_inf,
                    t_setup=t_setup, t_solve=t_solve,
                    iters=iters, conv=conv,
                    coarse_dim=P.shape[1],
                    include_S_in_e2e=True,
                )

            # ---- PyAMG baseline (auto coarse dim) ----
            if USE_PYAMG:
                t0 = time.perf_counter()
                P = pyamg_inference(A_csr, S_np)
                t_inf = time.perf_counter() - t0

                t_setup, t_solve, iters, conv = solve_with_P(A_csr, b, P, tol=TOL)

                push(
                    (N, "PyAMG_SA"),
                    t_A=t_A, t_S=t_S, t_inf=t_inf,
                    t_setup=t_setup, t_solve=t_solve,
                    iters=iters, conv=conv,
                    coarse_dim=P.shape[1],
                    include_S_in_e2e=False,  
                )

    # -----------------------
    # Summarize + save CSV  
    # -----------------------
    rows = []
    for (N, method), d in stats.items():
        K_vec = K_for_N(N)
        r_tgt = r_for_N(N)

        train_time = training_times.get(method, 0.0)

        rows.append({
            "N": N,
            "n": (N + 1) ** 2,
            "K_vectors": K_vec,
            "r_target": r_tgt,
            "Method": method,
            "Train Time (s)": train_time,

            "Coarse Dim (median)": float(np.median(d["coarse_dim"])),
            "Coarse Dim (mean)": float(np.mean(d["coarse_dim"])),

            "A Gen (ms)": 1000 * float(np.mean(d["t_A"])),
            "S Gen (ms)": 1000 * float(np.mean(d["t_S"])),

            "Inference (ms)": 1000 * float(np.mean(d["t_inf"])),
            "Setup (ms)": 1000 * float(np.mean(d["t_setup"])),
            "Solve-to-tol (ms)": 1000 * float(np.mean(d["t_solve_tol"])),

            "Pipeline (ms)": 1000 * float(np.mean(d["t_pipeline"])),
            "End-to-End (ms)": 1000 * float(np.mean(d["t_e2e"])),
            "End-to-End Std (ms)": 1000 * float(np.std(d["t_e2e"])),

            "Median Iters-to-tol": float(np.median(d["iters_tol"])),
            "Converged %": 100.0 * float(np.mean(d["converged"])),
        })

    df = pd.DataFrame(rows).sort_values(["N", "Method"])
    df["BaseMethod"] = df["Method"].astype(str).apply(lambda s: re.sub(r"_N=\d+", "", s))

    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/scalability_ruleK=N_r=N2_{pde_type}_tol{TOL}_S{TEST_SAMPLES}_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    # -----------------------
    # Plot
    # -----------------------
    metric = "End-to-End (ms)"
    plt.figure(figsize=(8, 4))

    for base in sorted(df["BaseMethod"].unique()):
        sub = df[df["BaseMethod"] == base].sort_values("N")
        x = sub["N"].to_numpy()
        y = sub[metric].to_numpy()

        plt.plot(x, y, marker="o", label=base)

        if metric == "End-to-End (ms)" and "End-to-End Std (ms)" in sub.columns:
            s = sub["End-to-End Std (ms)"].to_numpy()
            plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("N")
    plt.ylabel(metric)
    plt.title(f"Scalability (rule K=N, r=N/2): {metric} vs N")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("scalability_lines.pdf")
    plt.show()



    
    
