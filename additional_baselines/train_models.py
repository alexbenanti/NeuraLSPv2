# scripts/train_models.py
import time
import numpy as np
import torch
import torch.optim as optim

from src.ckpt_utils import normalize_pde_types, get_ckpt_path, save_ckpt, load_ckpt

from src.external_baselines import (
    GreenfeldStencilMLP,
    GreenfeldFrobeniusLoss,
    NeurKIttFNO,
    NeuralIFFactorNet,
    ProjectionLoss,
    grid_input_from_matrix,
    build_neuralif_graph,
    neuralif_sketched_loss,
    smallest_invariant_subspace_from_matrix,
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
N = 64
K_VECTORS = 72
RANKS = [36, 48, 64, 72]
RANK_MAX = max(RANKS)

TRAIN_EPOCHS = 500
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

CKPT_ROOT = "checkpoints"

PDE_TYPES_RAW = ["diffusion", "heat_equation", "wave_equation", "anisotropic", "screened_poisson"]
#PDE_TYPES_RAW = ["anisotropic", "screened_poisson"]
#PDE_TYPES_RAW = ["diffusion"]
PDE_TYPES = normalize_pde_types(PDE_TYPES_RAW, dedupe=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


def get_features(A_csr, S_np):
    return torch.FloatTensor(S_np).to(DEVICE)


def _loss_name(loss_fn):
    return getattr(loss_fn, "__name__", loss_fn.__class__.__name__)


def _model_output_rank(model) -> int:
    rank = getattr(model, "output_rank", None)
    if rank is None:
        raise AttributeError(f"Model {model.__class__.__name__} does not expose output_rank")
    return int(rank)


NEURKITT_K = 48


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


def train_model(model, optimizer, loss_fn, n_steps, pde_type: str, cfg: dict):
    model_kind = cfg["kind"]
    input_representation = cfg.get("input_representation", "summary")
    normalize_input = cfg.get("normalize_input", True)

    model.train()
    t0 = time.perf_counter()

    for step in range(n_steps):
        A_csr = generate_pde_data(N, pde_type=pde_type)

        if model_kind in {"mlp", "gnn"}:
            S_np = smooth_test_vectors(A_csr, num_vectors=K_VECTORS)
            perm = np.random.permutation(S_np.shape[1])
            S_np = S_np[:, perm]
            S_target = torch.FloatTensor(S_np).unsqueeze(0).to(DEVICE)
        else:
            S_np = None
            S_target = None

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

        elif model_kind == "greenfeld":
            x_grid = grid_input_from_matrix(
                A_csr,
                DEVICE,
                representation=input_representation,
                normalize=normalize_input,
            )
            P = model(x_grid)
            loss = loss_fn(P, A_csr)

        elif model_kind == "neurkitt":
            x_grid = grid_input_from_matrix(
                A_csr,
                DEVICE,
                representation=input_representation,
                normalize=normalize_input,
            )
            Q = model(x_grid)
            target_np = smallest_invariant_subspace_from_matrix(
                A_csr,
                target_rank=_model_output_rank(model),
            )
            S_target = torch.from_numpy(target_np).unsqueeze(0).to(DEVICE)
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

    # mlp_nested = ProlongationMLP2(input_dim, 128, 256, RANK_MAX, n_nodes, RANK_MAX).to(DEVICE)
    # models["MLP_Nested"] = {
    #     "model": mlp_nested,
    #     "loss": nested_lora_loss,
    #     "opt": optim.Adam(mlp_nested.parameters(), lr=LR),
    #     "kind": "mlp",
    #     "solver": "two_grid",
    #     "rank_sweep": True,
    #     "uses_smooth_vectors": True,
    # }

    # mlp_unnested = ProlongationMLP2(input_dim, 128, 256, RANK_MAX, n_nodes, RANK_MAX).to(DEVICE)
    # models["MLP_Unnested"] = {
    #     "model": mlp_unnested,
    #     "loss": subspace_loss,
    #     "opt": optim.Adam(mlp_unnested.parameters(), lr=LR),
    #     "kind": "mlp",
    #     "solver": "two_grid",
    #     "rank_sweep": True,
    #     "uses_smooth_vectors": True,
    # }

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

    greenfeld = GreenfeldStencilMLP(
        in_channels=9,
        hidden_dim=50,
        depth=5,
        output_rank=GREENFELD_PAPER_NC,
    ).to(DEVICE)
    models["Greenfeld2019_stencil45_frob"] = {
        "model": greenfeld,
        "loss": GreenfeldFrobeniusLoss(),
        "opt": optim.Adam(greenfeld.parameters(), lr=5e-4),
        "kind": "greenfeld",
        "solver": "two_grid",
        "rank_sweep": False,
        "paper_rank": GREENFELD_PAPER_NC,
        "uses_smooth_vectors": False,
        "input_representation": "stencil",
        "normalize_input": False,
    }

    # neurkitt = NeurKIttFNO(
    #     in_channels=7,
    #     width=48,
    #     modes1=16,
    #     modes2=16,
    #     num_layers=4,
    #     output_rank=NEURKITT_K,
    # ).to(DEVICE)
    # models["NeurKItt_FNO_k48_paperProjection"] = {
    #     "model": neurkitt,
    #     "loss": ProjectionLoss(),
    #     "opt": optim.Adam(neurkitt.parameters(), lr=5e-4),
    #     "kind": "neurkitt",
    #     "solver": "deflated_cg",
    #     "rank_sweep": False,
    #     "paper_rank": NEURKITT_K,
    #     "uses_smooth_vectors": False,
    #     "input_representation": "summary",
    #     "normalize_input": True,
    # }

    

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
                pde_key=pde_type,
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
                cfg=cfg,
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
