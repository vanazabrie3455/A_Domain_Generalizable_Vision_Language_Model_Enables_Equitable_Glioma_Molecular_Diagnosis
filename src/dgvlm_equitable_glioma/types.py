from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import torch
from torch import Tensor

TargetName = Literal["idh", "codeletion", "mgmt", "subtype"]


@dataclass(frozen=True)
class PatchRecord:
    slide_id: str
    domain: str
    path: Path
    x: int
    y: int
    level: int
    tissue_fraction: float
    blur_score: float


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    patient_id: str
    domain: str
    path: Path
    labels: dict[TargetName, int]
    available: frozenset[TargetName] = field(default_factory=frozenset)


@dataclass(frozen=True)
class FeatureRecord:
    slide_id: str
    domain: str
    features: Tensor
    text_embeddings: Tensor
    labels: dict[TargetName, int]


@dataclass(frozen=True)
class ModelSettings:
    feature_dimension: int = 512
    attention_dimension: int = 256
    hidden_dimension: int = 256
    dropout: float = 0.3
    fsdr_alpha: float = 0.3
    text_boost_beta: float = 0.5
    dcr_weight: float = 0.1
    entropy_weight: float = 0.001
    temperature: float = 0.07
    ema_momentum: float = 0.99


@dataclass(frozen=True)
class OptimizerSettings:
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    epochs: int = 200
    accumulation: int = 8
    patience: int = 20
    restart_period: int = 20
    restart_multiplier: int = 2


class ModelOutput(NamedTuple):
    logits: dict[TargetName, Tensor]
    representation: Tensor
    attention: Tensor
    text_scores: Tensor


class LossOutput(NamedTuple):
    total: Tensor
    task: Tensor
    consistency: Tensor
    entropy: Tensor
    components: dict[str, Tensor]


def move_feature_record(record: FeatureRecord, device: torch.device) -> FeatureRecord:
    return FeatureRecord(
        slide_id=record.slide_id,
        domain=record.domain,
        features=record.features.to(device),
        text_embeddings=record.text_embeddings.to(device),
        labels=record.labels,
    )
