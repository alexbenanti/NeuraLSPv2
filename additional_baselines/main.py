# scripts/main.py
import numpy as np
import torch

from src.ckpt_utils import get_ckpt_path, load_ckpt
from train_models import (
    build_models,
    CKPT_ROOT,
    N,
    K_VECTORS,
    RANKS,
    RANK_MAX,
    DEVICE,
    SEED,
    PDE_TYPES,
)
from rank_sweep import run_rank_sweep

np.random.seed(SEED)
torch.manual_seed(SEED)

TEST_SAMPLES = 100

SKIP_FACTORIZED_BASELINES = True


def drop_factorized_models(models):
    kept = {}
    removed = []

    for name, cfg in models.items():
        is_factorized = (name == "NeuralIF")
        if is_factorized:
            removed.append(name)
        else:
            kept[name] = cfg

    if removed:
        print(f"Skipping factorized baselines: {', '.join(removed)}")

    return kept

if __name__ == "__main__":
    for pde_type in PDE_TYPES:
        print(f"\n==============================")
        print(f"BENCHMARK for PDE: {pde_type}")
        print(f"==============================")

        models = build_models()
        if SKIP_FACTORIZED_BASELINES:
            models = drop_factorized_models(models)
        training_times = {}

        for name, cfg in models.items():
            ckpt_path = get_ckpt_path(
                root=CKPT_ROOT,
                model_name=name,
                pde_key=pde_type,
                N=N, K=K_VECTORS, R=RANK_MAX, seed=SEED,
            )
            meta = load_ckpt(ckpt_path, cfg["model"], optimizer=None, device=DEVICE)
            cfg["model"].eval()
            training_times[name] = float(meta.get("train_time_s", 0.0))
            print(f"  -> loaded {name} from {ckpt_path}")

        df = run_rank_sweep(
            models=models,
            training_times=training_times,
            ranks=RANKS,
            num_samples=TEST_SAMPLES,
            pde_type=pde_type,
        )

        df.to_csv(
            f"PERCENTILE_rank_sweep_{pde_type}_N{N}_K{K_VECTORS}_Rmax{RANK_MAX}.csv",
            index=False,
        )
        print(f"Saved CSV for {pde_type}")