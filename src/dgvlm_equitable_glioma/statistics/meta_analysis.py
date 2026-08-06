from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Heterogeneity:
    cochran_q: float
    degrees_of_freedom: int
    p_value: float
    i_squared: float
    tau_squared: float


@dataclass(frozen=True)
class RandomEffectsEstimate:
    estimate: float
    standard_error: float
    lower: float
    upper: float
    prediction_lower: float
    prediction_upper: float
    tau_squared: float


def fixed_effect_mean(effects: FloatArray, variances: FloatArray) -> tuple[float, FloatArray]:
    if bool(np.any(variances <= 0.0)):
        raise ValueError("variances must be positive")
    weights = 1.0 / variances
    estimate = float(np.sum(weights * effects) / np.sum(weights))
    return estimate, weights


def heterogeneity(effects: FloatArray, variances: FloatArray) -> Heterogeneity:
    if effects.shape != variances.shape or effects.ndim != 1:
        raise ValueError("effects and variances must be equal vectors")
    if len(effects) < 2:
        raise ValueError("heterogeneity requires at least two sites")
    estimate, weights = fixed_effect_mean(effects, variances)
    q = float(np.sum(weights * np.square(effects - estimate)))
    degrees = len(effects) - 1
    p_value = float(stats.chi2.sf(q, degrees))
    i_squared = max(0.0, (q - degrees) / q) * 100.0 if q > 0.0 else 0.0
    sum_weights = float(weights.sum())
    correction = sum_weights - float(np.square(weights).sum()) / sum_weights
    tau_squared = max(0.0, (q - degrees) / correction)
    return Heterogeneity(q, degrees, p_value, i_squared, tau_squared)


def random_effects(
    effects: FloatArray, variances: FloatArray, confidence: float = 0.95
) -> RandomEffectsEstimate:
    summary = heterogeneity(effects, variances)
    weights = 1.0 / (variances + summary.tau_squared)
    estimate = float(np.sum(weights * effects) / np.sum(weights))
    standard_error = float(np.sqrt(1.0 / np.sum(weights)))
    probability = 0.5 + confidence / 2.0
    critical = float(stats.t.ppf(probability, len(effects) - 1))
    lower = estimate - critical * standard_error
    upper = estimate + critical * standard_error
    prediction_error = float(np.sqrt(summary.tau_squared + standard_error**2))
    prediction_lower = estimate - critical * prediction_error
    prediction_upper = estimate + critical * prediction_error
    return RandomEffectsEstimate(
        estimate,
        standard_error,
        lower,
        upper,
        prediction_lower,
        prediction_upper,
        summary.tau_squared,
    )


def auc_variance(auc: float, positives: int, negatives: int) -> float:
    if positives <= 0 or negatives <= 0:
        raise ValueError("both outcome classes must be represented")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    numerator = auc * (1.0 - auc)
    numerator += (positives - 1) * (q1 - auc * auc)
    numerator += (negatives - 1) * (q2 - auc * auc)
    return numerator / (positives * negatives)


def hartung_knapp(
    effects: FloatArray,
    variances: FloatArray,
    confidence: float = 0.95,
) -> RandomEffectsEstimate:
    summary = heterogeneity(effects, variances)
    weights = 1.0 / (variances + summary.tau_squared)
    estimate = float(np.sum(weights * effects) / np.sum(weights))
    q_star = float(np.sum(weights * np.square(effects - estimate)) / (len(effects) - 1))
    standard_error = float(np.sqrt(q_star / np.sum(weights)))
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, len(effects) - 1))
    prediction_error = float(np.sqrt(summary.tau_squared + standard_error**2))
    return RandomEffectsEstimate(
        estimate=estimate,
        standard_error=standard_error,
        lower=estimate - critical * standard_error,
        upper=estimate + critical * standard_error,
        prediction_lower=estimate - critical * prediction_error,
        prediction_upper=estimate + critical * prediction_error,
        tau_squared=summary.tau_squared,
    )
