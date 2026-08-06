from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BinaryMetrics:
    auc_roc: float
    auc_pr: float
    weighted_f1: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    threshold: float
    kappa: float
    brier: float


@dataclass(frozen=True)
class MulticlassMetrics:
    macro_auc: float
    weighted_auc: float
    weighted_f1: float
    balanced_accuracy: float
    kappa: float


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    samples: FloatArray


def youden_threshold(labels: IntArray, probabilities: FloatArray) -> float:
    false_positive, true_positive, thresholds = roc_curve(labels, probabilities)
    index = int(np.argmax(true_positive - false_positive))
    return float(thresholds[index])


def binary_metrics(labels: IntArray, probabilities: FloatArray) -> BinaryMetrics:
    threshold = youden_threshold(labels, probabilities)
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    return BinaryMetrics(
        auc_roc=float(roc_auc_score(labels, probabilities)),
        auc_pr=float(average_precision_score(labels, probabilities)),
        weighted_f1=float(f1_score(labels, predictions, average="weighted")),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        threshold=threshold,
        kappa=float(cohen_kappa_score(labels, predictions)),
        brier=float(brier_score_loss(labels, probabilities)),
    )


def multiclass_metrics(labels: IntArray, probabilities: FloatArray) -> MulticlassMetrics:
    predictions = probabilities.argmax(axis=1)
    return MulticlassMetrics(
        macro_auc=float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
        weighted_auc=float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")
        ),
        weighted_f1=float(f1_score(labels, predictions, average="weighted")),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        kappa=float(cohen_kappa_score(labels, predictions)),
    )


def stratified_bootstrap(
    labels: IntArray,
    probabilities: FloatArray,
    statistic: Callable[[IntArray, FloatArray], float],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    generator = np.random.default_rng(seed)
    classes = np.unique(labels)
    class_indices = [np.flatnonzero(labels == value) for value in classes]
    values = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [generator.choice(indices, len(indices), replace=True) for indices in class_indices]
        )
        generator.shuffle(sampled)
        values[iteration] = statistic(labels[sampled], probabilities[sampled])
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=statistic(labels, probabilities),
        lower=float(np.quantile(values, alpha)),
        upper=float(np.quantile(values, 1.0 - alpha)),
        samples=values,
    )


def calibration_errors(
    labels: IntArray,
    probabilities: FloatArray,
    bins: int = 10,
) -> tuple[float, float]:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    expected = 0.0
    maximum = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (probabilities >= lower) & (probabilities < upper)
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
        expected += count / len(labels) * gap
        maximum = max(maximum, gap)
    return expected, maximum
