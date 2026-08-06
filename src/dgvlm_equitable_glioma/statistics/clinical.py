from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClinicalImpact:
    raw_gain: float
    false_positive_corrected_gain: float
    false_positive_burden: float
    number_needed_to_test: float


def raw_correct_gain(
    sensitivity_model: float,
    sensitivity_baseline: float,
    prevalence: float,
    population: int = 1000,
) -> float:
    return (sensitivity_model - sensitivity_baseline) * prevalence * population


def corrected_gain(
    sensitivity_model: float,
    specificity_model: float,
    sensitivity_baseline: float,
    specificity_baseline: float,
    prevalence: float,
    population: int = 1000,
) -> float:
    true_positive_gain = raw_correct_gain(
        sensitivity_model,
        sensitivity_baseline,
        prevalence,
        population,
    )
    model_false_positives = (1.0 - specificity_model) * (1.0 - prevalence) * population
    baseline_false_positives = (1.0 - specificity_baseline) * (1.0 - prevalence) * population
    return true_positive_gain - (model_false_positives - baseline_false_positives)


def clinical_impact(
    sensitivity_model: float,
    specificity_model: float,
    sensitivity_baseline: float,
    specificity_baseline: float,
    prevalence: float = 0.4,
    population: int = 1000,
) -> ClinicalImpact:
    raw = raw_correct_gain(sensitivity_model, sensitivity_baseline, prevalence, population)
    corrected = corrected_gain(
        sensitivity_model,
        specificity_model,
        sensitivity_baseline,
        specificity_baseline,
        prevalence,
        population,
    )
    false_positive_burden = (1.0 - specificity_model) * (1.0 - prevalence) * population
    number_needed = population / corrected if corrected > 0.0 else float("inf")
    return ClinicalImpact(raw, corrected, false_positive_burden, number_needed)


def impact_bootstrap(
    sensitivities: NDArray[np.float64],
    baseline: float,
    prevalence: float = 0.4,
    population: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    gains = (sensitivities - baseline) * prevalence * population
    alpha = (1.0 - confidence) / 2.0
    return (
        float(gains.mean()),
        float(np.quantile(gains, alpha)),
        float(np.quantile(gains, 1.0 - alpha)),
    )
