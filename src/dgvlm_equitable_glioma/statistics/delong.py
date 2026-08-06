from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DeLongResult:
    auc_first: float
    auc_second: float
    difference: float
    variance: float
    z_score: float
    p_value: float


def midranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    result = np.empty(len(values), dtype=np.float64)
    result[order] = ranks
    return result


def weighted_midranks(values: FloatArray, weights: FloatArray) -> FloatArray:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    ranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        left = cumulative[start - 1] if start > 0 else 0.0
        right = cumulative[end - 1]
        ranks[start:end] = 0.5 * (left + right)
        start = end
    result = np.empty(len(values), dtype=np.float64)
    result[order] = ranks
    return result


def covariance_matrix(values: FloatArray) -> FloatArray:
    covariance = np.cov(values, bias=False)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    return covariance


def fast_delong(
    predictions: FloatArray,
    positive_count: int,
) -> tuple[FloatArray, FloatArray]:
    classifiers, examples = predictions.shape
    negative_count = examples - positive_count
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    positive_ranks = np.empty((classifiers, positive_count), dtype=np.float64)
    negative_ranks = np.empty((classifiers, negative_count), dtype=np.float64)
    combined_ranks = np.empty((classifiers, examples), dtype=np.float64)
    for index in range(classifiers):
        positive_ranks[index] = midranks(positive[index])
        negative_ranks[index] = midranks(negative[index])
        combined_ranks[index] = midranks(predictions[index])
    aucs = combined_ranks[:, :positive_count].sum(axis=1)
    aucs = aucs / (positive_count * negative_count) - (positive_count + 1.0) / (
        2.0 * negative_count
    )
    positive_components = (combined_ranks[:, :positive_count] - positive_ranks) / negative_count
    negative_components = (
        1.0 - (combined_ranks[:, positive_count:] - negative_ranks) / positive_count
    )
    covariance = covariance_matrix(positive_components) / positive_count
    covariance += covariance_matrix(negative_components) / negative_count
    return aucs, covariance


def ordered_predictions(
    labels: IntArray,
    prediction_sets: list[FloatArray],
) -> tuple[FloatArray, int]:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("DeLong comparison requires binary labels")
    if any(len(predictions) != len(labels) for predictions in prediction_sets):
        raise ValueError("all predictions must align with labels")
    order = np.argsort(-labels)
    matrix = np.vstack([predictions[order] for predictions in prediction_sets])
    return matrix, int(labels.sum())


def delong_test(labels: IntArray, first: FloatArray, second: FloatArray) -> DeLongResult:
    matrix, positive_count = ordered_predictions(labels, [first, second])
    aucs, covariance = fast_delong(matrix, positive_count)
    contrast = np.asarray([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    difference = float(aucs[0] - aucs[1])
    if variance <= 0.0:
        z_score = float("inf") if difference != 0.0 else 0.0
        p_value = 0.0 if difference != 0.0 else 1.0
    else:
        z_score = difference / np.sqrt(variance)
        p_value = float(2.0 * stats.norm.sf(abs(z_score)))
    return DeLongResult(float(aucs[0]), float(aucs[1]), difference, variance, z_score, p_value)


def bonferroni(values: FloatArray, comparisons: int | None = None) -> FloatArray:
    count = len(values) if comparisons is None else comparisons
    return np.minimum(values * count, 1.0)


def holm(values: FloatArray) -> FloatArray:
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        value = min(1.0, float(values[index]) * (count - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted
