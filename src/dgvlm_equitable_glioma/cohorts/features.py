from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from dgvlm_equitable_glioma.cohorts.manifest import Manifest
from dgvlm_equitable_glioma.types import FeatureRecord, SlideRecord, TargetName


class FeatureDataset(Dataset[FeatureRecord]):
    def __init__(
        self,
        manifest: Manifest,
        feature_root: Path,
        prototypes: Tensor,
        maximum_patches: int | None = None,
        sample_patches: bool = True,
    ) -> None:
        self.records = manifest.records
        self.feature_root = feature_root
        self.prototypes = prototypes.float()
        self.maximum_patches = maximum_patches
        self.sample_patches = sample_patches

    def __len__(self) -> int:
        return len(self.records)

    def _feature_path(self, record: SlideRecord) -> Path:
        return self.feature_root / record.domain / f"{record.slide_id}.h5"

    def _load_features(self, record: SlideRecord) -> Tensor:
        path = self._feature_path(record)
        with h5py.File(path, "r") as handle:
            array = np.asarray(handle["features"], dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 512:
            raise ValueError(f"invalid feature shape for {record.slide_id}: {array.shape}")
        return torch.from_numpy(array)

    def _choose(self, features: Tensor) -> Tensor:
        maximum = self.maximum_patches
        if maximum is None or features.shape[0] <= maximum:
            return features
        if self.sample_patches:
            indices = torch.randperm(features.shape[0])[:maximum]
        else:
            indices = torch.linspace(0, features.shape[0] - 1, maximum).long()
        return features.index_select(0, indices)

    def __getitem__(self, index: int) -> FeatureRecord:
        record = self.records[index]
        features = self._choose(self._load_features(record))
        return FeatureRecord(
            slide_id=record.slide_id,
            domain=record.domain,
            features=features,
            text_embeddings=self.prototypes,
            labels=record.labels,
        )


def collate_slides(records: Sequence[FeatureRecord]) -> list[FeatureRecord]:
    return list(records)


def labels_for_target(records: Sequence[FeatureRecord], target: TargetName) -> Tensor:
    values = [record.labels[target] for record in records if target in record.labels]
    return torch.tensor(values, dtype=torch.long)


def domain_index(records: Sequence[FeatureRecord], vocabulary: Sequence[str]) -> Tensor:
    mapping = {name: index for index, name in enumerate(vocabulary)}
    return torch.tensor([mapping[record.domain] for record in records], dtype=torch.long)
