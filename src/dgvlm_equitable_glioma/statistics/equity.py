from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EquitySummary:
    mean: float
    standard_deviation: float
    coefficient_of_variation: float
    gini: float
    minimum_maximum_ratio: float
    range: float


def gini_coefficient(values: FloatArray) -> float:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty vector")
    if bool(np.any(values < 0)):
        raise ValueError("gini inputs must be nonnegative")
    total = float(values.sum())
    if total == 0.0:
        return 0.0
    ordered = np.sort(values)
    indices = np.arange(1, len(ordered) + 1, dtype=np.float64)
    numerator = float(np.sum((2.0 * indices - len(ordered) - 1.0) * ordered))
    return numerator / (len(ordered) * total)


def equity_summary(values: FloatArray) -> EquitySummary:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("site values must be a non-empty vector")
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    minimum = float(values.min())
    maximum = float(values.max())
    return EquitySummary(
        mean=mean,
        standard_deviation=standard_deviation,
        coefficient_of_variation=standard_deviation / mean,
        gini=gini_coefficient(values),
        minimum_maximum_ratio=minimum / maximum,
        range=maximum - minimum,
    )


def generalization_gap(in_domain_auc: float, external_auc: float) -> float:
    return in_domain_auc - external_auc


def generalization_ratio(in_domain_auc: float, external_auc: float) -> float:
    if in_domain_auc == 0.0:
        raise ZeroDivisionError("in-domain AUC must be nonzero")
    return external_auc / in_domain_auc


def tier_difference(values: FloatArray, tiers: NDArray[np.str_], first: str, second: str) -> float:
    first_values = values[tiers == first]
    second_values = values[tiers == second]
    if first_values.size == 0 or second_values.size == 0:
        raise ValueError("both tiers must contain observations")
    return float(first_values.mean() - second_values.mean())


def pairwise_site_gaps(values: FloatArray, names: list[str]) -> dict[tuple[str, str], float]:
    if len(values) != len(names):
        raise ValueError("site names and values must have equal length")
    result: dict[tuple[str, str], float] = {}
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            result[(names[left], names[right])] = float(values[left] - values[right])
    return result
