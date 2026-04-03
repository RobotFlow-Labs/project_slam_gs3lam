"""Dataset factory for GS3LAM."""

from __future__ import annotations

from anima_slam_gs3lam.config import DatasetPreset

from .replica import ReplicaSemanticDataset
from .scannet import ScannetSemanticDataset
from .tum import TUMSemanticDataset


DATASET_REGISTRY = {
    "replica": ReplicaSemanticDataset,
    "scannet": ScannetSemanticDataset,
    "tum": TUMSemanticDataset,
}


def build_dataset(name: str, config: DatasetPreset):
    try:
        dataset_cls = DATASET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}") from exc
    return dataset_cls(config)
