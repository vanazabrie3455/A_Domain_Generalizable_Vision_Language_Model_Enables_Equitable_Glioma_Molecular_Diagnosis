from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from dgvlm_equitable_glioma.statistics.classification import BinaryMetrics, binary_metrics
from dgvlm_equitable_glioma.types import FeatureRecord, ModelOutput, TargetName


@dataclass(frozen=True)
class PredictionRecord:
    slide_id: str
    domain: str
    target: TargetName
    label: int
    probabilities: NDArray[np.float64]


@dataclass(frozen=True)
class DomainEvaluation:
    domain: str
    target: TargetName
    samples: int
    metrics: BinaryMetrics


class ValidationRunner:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def _predict_record(self, record: FeatureRecord) -> ModelOutput:
        return self.model(
            record.features.to(self.device, non_blocking=True),
            record.text_embeddings.to(self.device, non_blocking=True),
        )

    @torch.no_grad()
    def predict(self, batches: Iterable[Sequence[FeatureRecord]]) -> list[PredictionRecord]:
        self.model.eval()
        predictions: list[PredictionRecord] = []
        for batch in batches:
            for record in batch:
                output = self._predict_record(record)
                for target, label in record.labels.items():
                    probabilities = torch.softmax(output.logits[target], dim=-1)
                    predictions.append(
                        PredictionRecord(
                            slide_id=record.slide_id,
                            domain=record.domain,
                            target=target,
                            label=label,
                            probabilities=probabilities.squeeze(0).cpu().double().numpy(),
                        )
                    )
        return predictions


def evaluate_binary(
    predictions: Sequence[PredictionRecord],
    target: TargetName,
    domain: str | None,
) -> DomainEvaluation:
    selected = [
        prediction
        for prediction in predictions
        if prediction.target == target and (domain is None or prediction.domain == domain)
    ]
    labels = np.asarray([prediction.label for prediction in selected], dtype=np.int64)
    probabilities = np.asarray(
        [prediction.probabilities[1] for prediction in selected],
        dtype=np.float64,
    )
    return DomainEvaluation(
        domain="pooled" if domain is None else domain,
        target=target,
        samples=len(selected),
        metrics=binary_metrics(labels, probabilities),
    )


def grouped_evaluations(
    predictions: Sequence[PredictionRecord],
) -> list[DomainEvaluation]:
    keys = sorted(
        {
            (prediction.domain, prediction.target)
            for prediction in predictions
            if len(prediction.probabilities) == 2
        }
    )
    return [evaluate_binary(predictions, target, domain) for domain, target in keys]
