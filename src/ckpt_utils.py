# src/ckpt_utils.py
from pathlib import Path
from typing import Union
import torch

def normalize_pde_types(pde_types, dedupe=True):
    """
    Keeps order; optionally de-dupes.
    If you WANT diffusion twice (oversampling), call with dedupe=False.
    """
    if not dedupe:
        return list(pde_types)
    out = []
    for p in pde_types:
        if p not in out:
            out.append(p)
    return out

def get_ckpt_path(root: Union[str,Path], model_name: str, pde_key: str, N: int, K: int, R: int, seed: int) -> Path:
    root = Path(root)
    ckpt_dir = root / pde_key / f"N{N}_K{K}_R{R}_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"{model_name}.pt"

def save_ckpt(path: Path, model, optimizer, meta: dict):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "meta": meta,
        },
        str(path),
    )

def load_ckpt(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("meta", {})
