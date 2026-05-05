import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from src.pdes import generate_pde_data, smooth_test_vectors
from src.model import ProlongationMLP,ProlongationMLP2, nested_lora_loss, subspace_loss

# ==============================================================================
# CONFIG
# ==============================================================================
N = 9
K_VECTORS = 32
RANK = 32
TRAIN_EPOCHS = 1000
TEST_SAMPLES = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pde_type = "screened_poisson"

# ranks you want to evaluate the *prefix* energy curve at
RANKS_TO_TEST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# For reproducibility (optional)
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================================================================
# HELPERS
# ==============================================================================

def plot_mean_with_error_bar(ranks, mean, std, label, marker="o"):
    plt.errorbar(
        ranks, 
        mean, 
        yerr=std, 
        label=label, 
        marker=marker,
        capsize=4,       # Adds horizontal caps to the error bars
        elinewidth=1.5,  # Thickness of the error bar lines
        fmt='-' + marker # Connects points with a line ('-o')
    )
def get_features(A_csr, S_np):
    # Your current choice: features = S
    return torch.tensor(S_np, dtype=torch.float32, device=DEVICE)

def train_model(model, optimizer, loss_fn, n_steps):
    """
    Trains on fresh PDE instances on the fly.
    """
    t0 = time.perf_counter()
    model.train()

    for _ in range(n_steps):
        #A_csr, S_np = generate_pde_data(N, k_vectors=K_VECTORS, pde_type=pde_type)

        A_csr = generate_pde_data(N, pde_type=pde_type)

        S_np = smooth_test_vectors(A_csr, num_vectors = K_VECTORS)

        # Random column permutation to avoid order leakage
        perm = np.random.permutation(S_np.shape[1])
        S_np = S_np[:, perm]

        x = get_features(A_csr, S_np).unsqueeze(0)              # (1, n, K)
        S_target = torch.tensor(S_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        Q = model(x)                                            # (1, n, RANK?) expected

        
        Q, _ = torch.linalg.qr(Q, mode="reduced")

        loss = loss_fn(Q, S_target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return time.perf_counter() - t0

def infer_Q(model, A_csr, S_np, rank=RANK):
    """
    Returns an (n, rank) orthonormal Q.
    """
    model.eval()
    with torch.no_grad():
        x = get_features(A_csr, S_np).unsqueeze(0)   # (1, n, K)
        Q = model(x).squeeze(0).detach().cpu().numpy()  # (n, rank)

    # Make sure the columns are orthonormal
    Q, _ = np.linalg.qr(Q, mode="reduced")
    return Q[:, :rank]

def energy_curve_from_Q(Q, S, ranks):
    """
    Computes captured energy fraction at each prefix rank r:
        ||Q_r^T S||_F^2 / ||S||_F^2

    Efficiently computed via cumulative row energies of Q^T S.
    """
    proj = Q.T @ S                    # (rank, K)
    row_energy = np.sum(proj**2, axis=1)  # (rank,)
    cum = np.cumsum(row_energy)          # (rank,)
    total = np.sum(S**2) + 1e-12

    out = []
    for r in ranks:
        r_eff = min(r, Q.shape[1])
        out.append(cum[r_eff - 1] / total)
    return np.array(out, dtype=np.float64)

def svd_energy_curve(S, ranks):
    """
    Oracle captured energy for SVD:
        sum_{i<=r} sigma_i^2 / sum_i sigma_i^2
    """
    _, sig, _ = np.linalg.svd(S, full_matrices=False)
    sig2 = sig**2
    denom = np.sum(sig2) + 1e-12
    cum = np.cumsum(sig2) / denom

    out = []
    for r in ranks:
        r_eff = min(r, len(cum))
        out.append(cum[r_eff - 1])
    return np.array(out, dtype=np.float64)

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    n_nodes = (N + 1) ** 2

    # --- Define models ---
    mlp_nested = ProlongationMLP(K_VECTORS, 128, RANK, n_nodes, RANK).to(DEVICE)
    mlp_unnested = ProlongationMLP(K_VECTORS, 128, RANK, n_nodes, RANK).to(DEVICE)

    opt_nested = optim.Adam(mlp_nested.parameters(), lr=LR)
    opt_unnested = optim.Adam(mlp_unnested.parameters(), lr=LR)

    # --- Train ONCE (offline) ---
    print(f"Training Nested for {TRAIN_EPOCHS} steps...")
    t_nested = train_model(mlp_nested, opt_nested, nested_lora_loss, TRAIN_EPOCHS)
    print(f"  done in {t_nested:.2f}s")

    print(f"Training Unnested for {TRAIN_EPOCHS} steps...")
    t_unnested = train_model(mlp_unnested, opt_unnested, subspace_loss, TRAIN_EPOCHS)
    print(f"  done in {t_unnested:.2f}s")

    # --- Evaluate across TEST_SAMPLES ---
    nested_mat = np.zeros((TEST_SAMPLES, len(RANKS_TO_TEST)), dtype=np.float64)
    unnested_mat = np.zeros((TEST_SAMPLES, len(RANKS_TO_TEST)), dtype=np.float64)
    svd_mat = np.zeros((TEST_SAMPLES, len(RANKS_TO_TEST)), dtype=np.float64)

    for i in range(TEST_SAMPLES):
        if i % 10 == 0:
            print(f"Evaluating sample {i}/{TEST_SAMPLES}...")

        

        A = generate_pde_data(N, pde_type=pde_type)

        S = smooth_test_vectors(A, num_vectors = K_VECTORS)

        Qn = infer_Q(mlp_nested, A, S, rank=RANK)
        Qu = infer_Q(mlp_unnested, A, S, rank=RANK)

        nested_mat[i, :] = energy_curve_from_Q(Qn, S, RANKS_TO_TEST)
        unnested_mat[i, :] = energy_curve_from_Q(Qu, S, RANKS_TO_TEST)
        svd_mat[i, :] = svd_energy_curve(S, RANKS_TO_TEST)


    nested_gap_mat = svd_mat - nested_mat
    unnested_gap_mat = svd_mat - unnested_mat 
    # --- Mean / Std ---
    mean_nested_gap = nested_gap_mat.mean(axis=0)
    std_nested_gap = nested_gap_mat.std(axis=0, ddof=1)

    mean_unnested_gap = unnested_gap_mat.mean(axis=0)
    std_unnested_gap = unnested_gap_mat.std(axis=0, ddof=1)

    

    # --- Plot with std error bars ---
    # --- Plot gap to SVD with std error bars ---
plt.figure(figsize=(10, 6))
plot_mean_with_error_bar(RANKS_TO_TEST, mean_nested_gap, std_nested_gap, "NLSS (ours)")
plot_mean_with_error_bar(RANKS_TO_TEST, mean_unnested_gap, std_unnested_gap, "Subspace Loss")

plt.xlabel("Rank $r$")
plt.ylabel(r"Energy gap to SVD: $E_{\mathrm{svd}}(r) - E_{\mathrm{model}}(r)$")
#plt.title(f"Captured Energy Gap to Ground Truth Truncated SVD vs Rank (mean ± std)\n{pde_type.capitalize()} Equation")
plt.title(f"Captured Energy Gap to Ground Truth Truncated SVD vs Rank (mean ± std)\n Screened Poisson Equation")
plt.grid(True, ls="--", alpha=0.4)
plt.grid(True, ls="--", alpha=0.4)
plt.legend()

# Reverse x-axis so it goes from high -> low left to right
plt.gca().invert_xaxis()

# Optional: auto y-limits with a little padding
ymin = min((mean_nested_gap - std_nested_gap).min(),
           (mean_unnested_gap - std_unnested_gap).min())
ymax = max((mean_nested_gap + std_nested_gap).max(),
           (mean_unnested_gap + std_unnested_gap).max())
pad = 0.05 * (ymax - ymin + 1e-12)
plt.ylim(ymin - pad, ymax + pad)

plt.tight_layout()

# --- SAVE AS PDF ---
plt.savefig(f"{pde_type}_energy_gap_to_svd.pdf", format="pdf", bbox_inches="tight")
plt.show()
