"""Shared runtime helpers for reproducible DEG2MOL entry points."""

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PathLike = Union[str, Path]


def project_path(path: PathLike) -> Path:
    """Resolve a user path, anchoring relative paths at the repository root."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def require_path(path: PathLike, description: str, directory: bool = False) -> Path:
    """Resolve and validate a required file or directory."""
    resolved = project_path(path)
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(
            f"Missing {description} {kind}: {resolved}\n"
            "Download and extract the external assets as described in README.md."
        )
    return resolved


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA when available, or validate an explicit device request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{requested}' was requested, but CUDA is unavailable.")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: PathLike, device: torch.device, description: str) -> Any:
    """Load a trusted local PyTorch checkpoint with a useful missing-file error."""
    checkpoint_path = require_path(path, description)
    return torch.load(str(checkpoint_path), map_location=device)


def checkpoint_state_dict(checkpoint: Any, description: str) -> Mapping:
    """Accept both a raw state_dict and a checkpoint wrapping model_state_dict."""
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{description} does not contain a valid PyTorch state_dict.")
    return state_dict


def load_model_weights(
    model: torch.nn.Module,
    path: PathLike,
    device: torch.device,
    description: str,
) -> Any:
    """Load checkpoint weights into a model and return the complete checkpoint."""
    checkpoint = load_checkpoint(path, device, description)
    model.load_state_dict(checkpoint_state_dict(checkpoint, description))
    return checkpoint


def load_gene_order(path: PathLike) -> List[str]:
    """Load the ordered DEG feature names from a text file or the first CSV column."""
    gene_path = require_path(path, "gene-order")
    if gene_path.suffix.lower() in {".txt", ".tsv"}:
        genes = [line.strip() for line in gene_path.read_text().splitlines() if line.strip()]
    else:
        first_column = pd.read_csv(gene_path, usecols=[0]).iloc[:, 0]
        genes = first_column.dropna().astype(str).tolist()
    if not genes:
        raise ValueError(f"Gene-order file is empty: {gene_path}")
    if len(set(genes)) != len(genes):
        raise ValueError(f"Gene-order file contains duplicate genes: {gene_path}")
    return genes


def ordered_deg_values(deg_df: pd.DataFrame, genes: List[str], split_name: str) -> np.ndarray:
    """Validate and return DEG values in the checkpoint's expected gene order."""
    missing = [gene for gene in genes if gene not in deg_df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{split_name} is missing {len(missing)} required genes. "
            f"First missing genes: {preview}"
        )
    return deg_df.loc[:, genes].to_numpy(dtype=np.float32, copy=True)


def load_scafvae(
    checkpoint_path: PathLike,
    device: torch.device,
    need_dataset_args: bool = False,
) -> Tuple[torch.nn.Module, Optional[Any]]:
    """Instantiate ScafVAE and load its externally distributed checkpoint."""
    try:
        from ScafVAE.model.main_layers import ScafVAEBase
    except ImportError as exc:
        raise ImportError(
            "ScafVAE is unavailable. Initialize submodules and install "
            "third_party/ScafVAE as described in README.md."
        ) from exc

    model = ScafVAEBase().to(device)
    checkpoint = load_model_weights(
        model,
        checkpoint_path,
        device,
        "ScafVAE checkpoint",
    )
    model.eval()

    dataset_args = checkpoint.get("args") if isinstance(checkpoint, Mapping) else None
    if need_dataset_args:
        if dataset_args is None:
            raise KeyError("ScafVAE checkpoint is missing the dataset configuration in 'args'.")
        dataset_args.is_main_process = True
        dataset_args.rand_inp = False
        dataset_args.n_batch = -1
        dataset_args.persistent_workers = False

    return model, dataset_args


def empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
