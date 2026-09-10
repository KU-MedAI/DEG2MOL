"""DEG table loading and validation shared by testing and inference."""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.runtime import ordered_deg_values, project_path, require_path


class DEGTableDataset(Dataset):
    """An index-preserving DEG dataset in the checkpoint's gene order."""

    def __init__(self, frame: pd.DataFrame, genes: List[str], split_name: str):
        if "cmap_name" not in frame.columns:
            raise ValueError(f"{split_name} must contain a 'cmap_name' column.")
        self.values = ordered_deg_values(frame, genes, split_name)
        self.names = frame["cmap_name"].astype(str).tolist()

    def __getitem__(self, index: int):
        return torch.from_numpy(self.values[index]), self.names[index]

    def __len__(self) -> int:
        return len(self.values)


def read_deg_table(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read a CSV or Feather DEG table without modifying the source file."""
    table_path = require_path(path, "DEG table")
    suffix = table_path.suffix.lower()
    if suffix == ".feather":
        frame = pd.read_feather(table_path)
    elif suffix == ".csv":
        frame = pd.read_csv(table_path)
    else:
        raise ValueError(f"Unsupported DEG table format: {table_path} (use .csv or .feather)")
    if max_rows is not None:
        frame = frame.head(max_rows).copy()
    return frame


def create_deg_loader(
    frame: pd.DataFrame,
    genes: List[str],
    split_name: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = DEGTableDataset(frame, genes, split_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_deg,
        pin_memory=torch.cuda.is_available(),
    )


def collate_deg(batch) -> Tuple[torch.Tensor, List[str]]:
    tensors = torch.stack([item[0] for item in batch], dim=0)
    names = [item[1] for item in batch]
    return tensors, names


def split_data_root(split: str, override: Optional[str] = None) -> Path:
    """Resolve the external random/scaffold split directory."""
    return project_path(override or f"data/splits/{split}")
