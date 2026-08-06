from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from dgvlm_equitable_glioma.statistics.classification import BinaryMetrics, binary_metrics

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


@dataclass(frozen=True)
class SubgroupResult:
    attribute: str
    subgroup: str
    samples: int
    positives: int
    negatives: int
    metrics: BinaryMetrics


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    iterations: int
    null_mean: float
    null_standard_deviation: float


def evaluate_subgroups(
    labels: IntArray,
    probabilities: FloatArray,
    groups: StringArray,
    attribute: str,
    minimum_samples: int = 10,
) -> list[SubgroupResult]:
    if not (labels.shape == probabilities.shape == groups.shape):
        raise ValueError("labels, probabilities, and groups must align")
    results: list[SubgroupResult] = []
    for group in sorted(np.unique(groups)):
        mask = groups == group
        selected_labels = labels[mask]
        if len(selected_labels) < minimum_samples or len(np.unique(selected_labels)) < 2:
            continue
        results.append(
            SubgroupResult(
                attribute=attribute,
                subgroup=str(group),
                samples=int(mask.sum()),
                positives=int(selected_labels.sum()),
                negatives=int(len(selected_labels) - selected_labels.sum()),
                metrics=binary_metrics(selected_labels, probabilities[mask]),
            )
        )
    return results


def maximum_auc_gap(results: Sequence[SubgroupResult]) -> float:
    if not results:
        raise ValueError("subgroup results cannot be empty")
    values = [result.metrics.auc_roc for result in results]
    return max(values) - min(values)


def permutation_gap_test(
    labels: IntArray,
    probabilities: FloatArray,
    groups: StringArray,
    iterations: int = 1000,
    seed: int = 42,
) -> PermutationResult:
    observed_results = evaluate_subgroups(labels, probabilities, groups, "group", minimum_samples=1)
    observed = maximum_auc_gap(observed_results)
    generator = np.random.default_rng(seed)
    null = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        permuted = generator.permutation(groups)
        results = evaluate_subgroups(labels, probabilities, permuted, "group", minimum_samples=1)
        null[iteration] = maximum_auc_gap(results)
    p_value = (float(np.sum(null >= observed)) + 1.0) / (iterations + 1.0)
    return PermutationResult(
        observed=observed,
        p_value=p_value,
        iterations=iterations,
        null_mean=float(null.mean()),
        null_standard_deviation=float(null.std(ddof=1)),
    )


def stratified_difference(
    values: FloatArray,
    groups: StringArray,
    strata: StringArray,
    first: str,
    second: str,
) -> tuple[float, float]:
    if not (values.shape == groups.shape == strata.shape):
        raise ValueError("values, groups, and strata must align")
    differences = []
    variances = []
    for stratum in np.unique(strata):
        mask = strata == stratum
        first_values = values[mask & (groups == first)]
        second_values = values[mask & (groups == second)]
        if len(first_values) < 2 or len(second_values) < 2:
            continue
        differences.append(float(first_values.mean() - second_values.mean()))
        variances.append(float(first_values.var(ddof=1) / len(first_values)))
        variances[-1] += float(second_values.var(ddof=1) / len(second_values))
    if not differences:
        raise ValueError("no stratum contains both groups")
    weights = 1.0 / np.asarray(variances)
    estimate = float(np.sum(weights * differences) / weights.sum())
    standard_error = float(np.sqrt(1.0 / weights.sum()))
    return estimate, standard_error


def interaction_test(
    outcomes: FloatArray,
    methods: StringArray,
    sites: StringArray,
) -> tuple[float, float]:
    if not (outcomes.shape == methods.shape == sites.shape):
        raise ValueError("outcomes, methods, and sites must align")
    method_levels = sorted(np.unique(methods))
    site_levels = sorted(np.unique(sites))
    grand_mean = outcomes.mean()
    residual_sum = 0.0
    interaction_sum = 0.0
    cells = 0
    for method in method_levels:
        method_mean = outcomes[methods == method].mean()
        for site in site_levels:
            mask = (methods == method) & (sites == site)
            if not mask.any():
                continue
            site_mean = outcomes[sites == site].mean()
            cell_mean = outcomes[mask].mean()
            interaction = cell_mean - method_mean - site_mean + grand_mean
            interaction_sum += int(mask.sum()) * interaction**2
            residual_sum += float(np.sum(np.square(outcomes[mask] - cell_mean)))
            cells += 1
    interaction_degrees = (len(method_levels) - 1) * (len(site_levels) - 1)
    residual_degrees = len(outcomes) - cells
    if interaction_degrees <= 0 or residual_degrees <= 0:
        raise ValueError("insufficient observations for interaction test")
    statistic = (interaction_sum / interaction_degrees) / (residual_sum / residual_degrees)
    p_value = float(stats.f.sf(statistic, interaction_degrees, residual_degrees))
    return float(statistic), p_value
