import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dgvlm_equitable_glioma.statistics.classification import BinaryMetrics, BootstrapInterval
from dgvlm_equitable_glioma.statistics.equity import EquitySummary, equity_summary
from dgvlm_equitable_glioma.statistics.meta_analysis import Heterogeneity, heterogeneity

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SeedResult:
    method: str
    domain: str
    target: str
    seed: int
    metrics: BinaryMetrics
    samples: int


@dataclass(frozen=True)
class AggregateResult:
    method: str
    domain: str
    target: str
    seeds: int
    samples: int
    mean: Mapping[str, float]
    standard_deviation: Mapping[str, float]


@dataclass(frozen=True)
class CrossSiteResult:
    method: str
    target: str
    site_auc: Mapping[str, float]
    equity: EquitySummary
    heterogeneity: Heterogeneity


METRIC_NAMES = (
    "auc_roc",
    "auc_pr",
    "weighted_f1",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "threshold",
    "kappa",
    "brier",
)


def aggregate_seed_results(results: Sequence[SeedResult]) -> list[AggregateResult]:
    groups: dict[tuple[str, str, str], list[SeedResult]] = {}
    for result in results:
        key = (result.method, result.domain, result.target)
        groups.setdefault(key, []).append(result)
    aggregates: list[AggregateResult] = []
    for key, records in sorted(groups.items()):
        values = {
            name: np.asarray(
                [getattr(record.metrics, name) for record in records], dtype=np.float64
            )
            for name in METRIC_NAMES
        }
        aggregates.append(
            AggregateResult(
                method=key[0],
                domain=key[1],
                target=key[2],
                seeds=len(records),
                samples=max(record.samples for record in records),
                mean={name: float(array.mean()) for name, array in values.items()},
                standard_deviation={
                    name: float(array.std(ddof=1)) if len(array) > 1 else 0.0
                    for name, array in values.items()
                },
            )
        )
    return aggregates


def cross_site_result(
    method: str,
    target: str,
    site_auc: Mapping[str, float],
    site_variance: Mapping[str, float],
) -> CrossSiteResult:
    sites = sorted(site_auc)
    if sites != sorted(site_variance):
        raise ValueError("site AUC and variance keys must match")
    effects = np.asarray([site_auc[site] for site in sites], dtype=np.float64)
    variances = np.asarray([site_variance[site] for site in sites], dtype=np.float64)
    return CrossSiteResult(
        method=method,
        target=target,
        site_auc={site: site_auc[site] for site in sites},
        equity=equity_summary(effects),
        heterogeneity=heterogeneity(effects, variances),
    )


def atomic_json(path: Path, content: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_seed_results(path: Path, results: Sequence[SeedResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = ["method", "domain", "target", "seed", "samples", *METRIC_NAMES]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "method": result.method,
                    "domain": result.domain,
                    "target": result.target,
                    "seed": result.seed,
                    "samples": result.samples,
                    **asdict(result.metrics),
                }
            )
    os.replace(temporary, path)


def write_aggregates(path: Path, results: Sequence[AggregateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = ["method", "domain", "target", "seeds", "samples"]
    for name in METRIC_NAMES:
        fieldnames.extend([f"{name}_mean", f"{name}_sd"])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: dict[str, object] = {
                "method": result.method,
                "domain": result.domain,
                "target": result.target,
                "seeds": result.seeds,
                "samples": result.samples,
            }
            for name in METRIC_NAMES:
                row[f"{name}_mean"] = result.mean[name]
                row[f"{name}_sd"] = result.standard_deviation[name]
            writer.writerow(row)
    os.replace(temporary, path)


def interval_dictionary(interval: BootstrapInterval) -> dict[str, float]:
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
    }


def result_matrix(
    results: Sequence[AggregateResult],
    metric: str = "auc_roc",
) -> tuple[list[str], list[str], FloatArray]:
    if metric not in METRIC_NAMES:
        raise ValueError(f"unknown metric: {metric}")
    methods = sorted({result.method for result in results})
    domains = sorted({result.domain for result in results})
    matrix = np.full((len(methods), len(domains)), np.nan, dtype=np.float64)
    method_index = {method: index for index, method in enumerate(methods)}
    domain_index = {domain: index for index, domain in enumerate(domains)}
    for result in results:
        matrix[method_index[result.method], domain_index[result.domain]] = result.mean[metric]
    return methods, domains, matrix


def rank_methods(matrix: FloatArray) -> FloatArray:
    ranks = np.full_like(matrix, np.nan)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        valid = np.flatnonzero(~np.isnan(values))
        order = valid[np.argsort(-values[valid])]
        for rank, index in enumerate(order, start=1):
            ranks[index, column] = rank
    return ranks


def mean_ranks(matrix: FloatArray) -> FloatArray:
    return np.nanmean(rank_methods(matrix), axis=1)


def read_seed_results(path: Path) -> list[SeedResult]:
    results: list[SeedResult] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metrics = BinaryMetrics(**{name: float(row[name]) for name in METRIC_NAMES})
            results.append(
                SeedResult(
                    method=row["method"],
                    domain=row["domain"],
                    target=row["target"],
                    seed=int(row["seed"]),
                    samples=int(row["samples"]),
                    metrics=metrics,
                )
            )
    return results
