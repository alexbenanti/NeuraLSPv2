# scripts/train_models.py
import time
import numpy as np
import torch
import torch.optim as optim

from src.ckpt_utils import normalize_pde_types, get_ckpt_path, save_ckpt, load_ckpt

from src.external_baselines import (
    GreenfeldStencilMLP,
    NeurKIttFNO,
    NeuralIFFactorNet,
    ProjectionLoss,
    grid_input_from_matrix,
    build_neuralif_graph,
    neuralif_sketched_loss,
)

from src.precorrector_baseline import (
    PreCorrectorIC0,
    precorrector_ic0_training_loss,
)

from src.model import (
    ProlongationMLP2,
    nested_lora_loss,
    subspace_loss,
    error_propagation_loss,
)

from src.pdes import generate_pde_data, smooth_test_vectors

# Optional GNN
try:
    from src.gnn_baseline import AMG_GNN
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# -----------------------
# CONFIG
# -----------------------

#must change N globally if changing the problem size

N = 8
K_VECTORS = 16
#RANKS = [2,4,6,8,10,12,14,20,24,26,28,30,32,36,40,44,48,52,56,60,64,68,72]
RANKS = [8]
RANK_MAX = max(RANKS)

TRAIN_EPOCHS = 1000
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

CKPT_ROOT = "checkpoints"

PDE_TYPES_RAW = ["diffusion", "anisotropic", "screened_poisson", "heat_equation", "wave_equation"]
PDE_TYPES = normalize_pde_types(PDE_TYPES_RAW, dedupe=True)  # set dedupe=False to oversample diffusion

np.random.seed(SEED)
torch.manual_seed(SEED)


def get_features(A_csr, S_np):
    return torch.FloatTensor(S_np).to(DEVICE)

def _loss_name(loss_fn):
    return getattr(loss_fn, "__name__", loss_fn.__class__.__name__)

NEURKITT_K = 36

def greenfeld_paper_nc_from_N(N: int) -> int:
    """
    Greenfeld 2019:
    fine grid has (N+1) x (N+1) nodal unknowns in this codebase,
    and the coarse grid skips every other mesh point.
    """
    fine_side = N + 1
    coarse_side = fine_side // 2 + 1
    return coarse_side * coarse_side

GREENFELD_PAPER_NC = greenfeld_paper_nc_from_N(N)

def train_model(model, optimizer, loss_fn, n_steps, pde_type: str, model_kind: str):
    model.train()
    t0 = time.perf_counter()

    for step in range(n_steps):
        A_csr = generate_pde_data(N, pde_type=pde_type)
        S_np = smooth_test_vectors(A_csr, num_vectors=K_VECTORS)

        perm = np.random.permutation(S_np.shape[1])
        S_np = S_np[:, perm]
        S_target = torch.FloatTensor(S_np).unsqueeze(0).to(DEVICE)

        if model_kind == "mlp":
            x = get_features(A_csr, S_np).unsqueeze(0)
            Q = model(x)
            loss = loss_fn(Q, S_target)

        elif model_kind == "gnn":
            x = get_features(A_csr, S_np)
            coo = A_csr.tocoo()
            edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=DEVICE)
            edge_attr = torch.tensor(coo.data, dtype=torch.float, device=DEVICE).unsqueeze(1)
            Q = model(x, edge_index, edge_attr).unsqueeze(0)
            loss = loss_fn(Q, A_csr)

        elif model_kind in {"greenfeld", "neurkitt"}:
            x_grid = grid_input_from_matrix(A_csr, DEVICE)
            Q = model(x_grid)
            loss = loss_fn(Q, S_target)

        elif model_kind == "neuralif":
            graph = build_neuralif_graph(A_csr, DEVICE)
            edge_values = model(graph.node_x, graph.edge_index, graph.edge_attr, graph.diag_mask)
            loss = neuralif_sketched_loss(graph.edge_index, edge_values, graph.size, A_csr, DEVICE)

        elif model_kind == "precorrector":
            loss = precorrector_ic0_training_loss(model, A_csr, DEVICE)

        else:
            raise ValueError(f"Unknown model kind: {model_kind}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 100 == 0:
            print(f"    [{pde_type}] {model_kind} step {step+1}/{n_steps} loss={loss.item():.6f}")

    return time.perf_counter() - t0

def build_models():
    input_dim = K_VECTORS
    n_nodes = (N + 1) ** 2

    models = {}

    mlp_nested = ProlongationMLP2(input_dim, 128, 256, RANK_MAX, n_nodes, RANK_MAX).to(DEVICE)
    models["MLP_Nested"] = {
        "model": mlp_nested,
        "loss": nested_lora_loss,
        "opt": optim.Adam(mlp_nested.parameters(), lr=LR),
        "kind": "mlp",
        "solver": "two_grid",
        "rank_sweep": True,
        "uses_smooth_vectors": True,
    }

    mlp_unnested = ProlongationMLP2(input_dim, 128, 256, RANK_MAX, n_nodes, RANK_MAX).to(DEVICE)
    models["MLP_Unnested"] = {
        "model": mlp_unnested,
        "loss": subspace_loss,
        "opt": optim.Adam(mlp_unnested.parameters(), lr=LR),
        "kind": "mlp",
        "solver": "two_grid",
        "rank_sweep": True,
        "uses_smooth_vectors": True,
    }

    # if GNN_AVAILABLE:
    #     gnn = AMG_GNN(input_node_dim=input_dim, output_dim=RANK_MAX, hidden_dim=488, num_layers=5).to(DEVICE)
    #     models["GNN"] = {
    #         "model": gnn,
    #         "loss": error_propagation_loss,
    #         "opt": optim.Adam(gnn.parameters(), lr=1e-4),
    #         "kind": "gnn",
    #         "solver": "two_grid",
    #         "rank_sweep": True,
    #         "uses_smooth_vectors": True,
    #         "gnn_do_qr": False,
    #     }
 
    # precor_ic0 = PreCorrectorIC0(hidden_dim=16, num_rounds=5).to(DEVICE)
    # models["PreCorrector_IC0"] = {
    #     "model": precor_ic0,
    #     "loss": precorrector_ic0_training_loss,
    #     "opt": optim.Adam(precor_ic0.parameters(), lr=1e-3),
    #     "kind": "precorrector",
    #     "solver": "factorized",
    #     "rank_sweep": False,
    #     "paper_rank": "IC(0)",
    #     "uses_smooth_vectors": False,
    # }

    # neuralif = NeuralIFFactorNet(
    #     node_in_dim=7,
    #     edge_in_dim=1,
    #     hidden_dim=64,
    #     message_passing_steps=3,
    # ).to(DEVICE)
    # models["NeuralIF"] = {
    #     "model": neuralif,
    #     "loss": neuralif_sketched_loss,
    #     "opt": optim.Adam(neuralif.parameters(), lr=5e-4),
    #     "kind": "neuralif",
    #     "solver": "factorized",
    #     "rank_sweep": False,
    #     "uses_smooth_vectors": False,
    # }

    return models

if __name__ == "__main__":
    for pde_type in PDE_TYPES:
        print(f"\n==============================")
        print(f"TRAIN/LOAD for PDE: {pde_type}")
        print(f"==============================")

        models = build_models()

        for name, cfg in models.items():
            ckpt_path = get_ckpt_path(
                root=CKPT_ROOT,
                model_name=name,
                pde_key=pde_type,   # <-- per-PDE checkpoint bucket
                N=N, K=K_VECTORS, R=RANK_MAX, seed=SEED,
            )

            if ckpt_path.exists():
                meta = load_ckpt(ckpt_path, cfg["model"], cfg["opt"], device=DEVICE)
                cfg["model"].eval()
                print(f"  -> loaded {name} from {ckpt_path} (train_time={meta.get('train_time_s', 0):.2f}s)")
                continue

            print(f"  -> training {name} for PDE={pde_type}")
            t_train = train_model(
                cfg["model"],
                cfg["opt"],
                cfg["loss"],
                n_steps=TRAIN_EPOCHS,
                pde_type=pde_type,
                model_kind=cfg["kind"],
            )

            meta = dict(
                pde_type=pde_type,
                N=N,
                K_VECTORS=K_VECTORS,
                RANK_MAX=RANK_MAX,
                SEED=SEED,
                model_name=name,
                model_class=cfg["model"].__class__.__name__,
                loss=_loss_name(cfg["loss"]),
                train_time_s=float(t_train),
            )
            save_ckpt(ckpt_path, cfg["model"], cfg["opt"], meta)
            print(f"  -> saved {name} to {ckpt_path} (train_time={t_train:.2f}s)")
