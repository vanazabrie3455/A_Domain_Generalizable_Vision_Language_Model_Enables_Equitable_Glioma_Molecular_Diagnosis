from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class McNemarResult:
    first_only_correct: int
    second_only_correct: int
    statistic: float
    p_value: float
    exact: bool


def mcnemar(
    labels: IntArray,
    first_predictions: IntArray,
    second_predictions: IntArray,
    exact_threshold: int = 25,
) -> McNemarResult:
    if not (labels.shape == first_predictions.shape == second_predictions.shape):
        raise ValueError("labels and predictions must have equal shapes")
    first_correct = first_predictions == labels
    second_correct = second_predictions == labels
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    discordant = first_only + second_only
    if discordant == 0:
        return McNemarResult(first_only, second_only, 0.0, 1.0, True)
    if discordant < exact_threshold:
        result = stats.binomtest(
            min(first_only, second_only), discordant, 0.5, alternative="two-sided"
        )
        return McNemarResult(first_only, second_only, float("nan"), float(result.pvalue), True)
    statistic = (abs(first_only - second_only) - 1.0) ** 2 / discordant
    p_value = float(stats.chi2.sf(statistic, 1))
    return McNemarResult(first_only, second_only, statistic, p_value, False)


@dataclass(frozen=True)
class OddsRatio:
    estimate: float
    lower: float
    upper: float
    p_value: float


def odds_ratio(table: NDArray[np.int64], confidence: float = 0.95) -> OddsRatio:
    if table.shape != (2, 2):
        raise ValueError("odds ratio requires a 2 by 2 table")
    corrected = table.astype(np.float64)
    if bool(np.any(corrected == 0.0)):
        corrected += 0.5
    first, second, third, fourth = corrected.ravel()
    estimate = first * fourth / (second * third)
    standard_error = np.sqrt(1.0 / first + 1.0 / second + 1.0 / third + 1.0 / fourth)
    critical = stats.norm.ppf(0.5 + confidence / 2.0)
    lower = np.exp(np.log(estimate) - critical * standard_error)
    upper = np.exp(np.log(estimate) + critical * standard_error)
    _, p_value = stats.fisher_exact(table)
    return OddsRatio(float(estimate), float(lower), float(upper), float(p_value))


def practical_significance(first_auc: float, second_auc: float, minimum: float = 0.03) -> bool:
    return first_auc - second_auc >= minimum
