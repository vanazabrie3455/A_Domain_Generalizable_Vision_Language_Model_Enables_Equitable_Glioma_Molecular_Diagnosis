import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dgvlm_equitable_glioma.cohorts.features import FeatureDataset, collate_slides
from dgvlm_equitable_glioma.cohorts.manifest import load_manifest
from dgvlm_equitable_glioma.cohorts.partitions import LeaveOneDomainOut
from dgvlm_equitable_glioma.configuration import (
    flattened,
    load_configuration,
    model_settings,
    optimizer_settings,
)
from dgvlm_equitable_glioma.learning.engine import EarlyStopping, ExperimentEngine, set_seed
from dgvlm_equitable_glioma.learning.objective import MultiTaskDomainObjective
from dgvlm_equitable_glioma.learning.validation import ValidationRunner, evaluate_binary
from dgvlm_equitable_glioma.representation.model import DomainGeneralizableVLM

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="dgvlm-train")
    result.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--features", type=Path, required=True)
    result.add_argument("--prototypes", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--held-out", type=str)
    result.add_argument("--validation-fold", type=int, default=0)
    result.add_argument("override", nargs="*")
    return result


def load_prototypes(path: Path) -> torch.Tensor:
    prototypes = torch.load(path, map_location="cpu")
    if not isinstance(prototypes, torch.Tensor):
        raise TypeError("prototype file must contain a tensor")
    if prototypes.ndim != 2 or prototypes.shape != (9, 512):
        raise ValueError(f"prototype tensor must have shape (9, 512), received {prototypes.shape}")
    return prototypes.float()


def initialize_profile(dataset: FeatureDataset) -> torch.Tensor:
    domain_sums: dict[str, torch.Tensor] = {}
    domain_counts: dict[str, int] = {}
    for index in range(len(dataset)):
        record = dataset[index]
        mean = record.features.mean(dim=0)
        if record.domain not in domain_sums:
            domain_sums[record.domain] = torch.zeros_like(mean)
            domain_counts[record.domain] = 0
        domain_sums[record.domain] += mean
        domain_counts[record.domain] += 1
    means = torch.stack(
        [domain_sums[domain] / domain_counts[domain] for domain in sorted(domain_sums)]
    )
    return means.std(dim=0, unbiased=True)


def run(arguments: argparse.Namespace) -> None:
    configuration = load_configuration(arguments.config, arguments.override)
    values = flattened(configuration)
    seed = int(values["seed"])
    set_seed(seed)
    manifest = load_manifest(arguments.manifest)
    held_out = arguments.held_out or str(values["held_out"])
    fold = LeaveOneDomainOut(
        manifest,
        validation_folds=int(values["validation_folds"]),
        seed=seed,
    ).split(held_out, arguments.validation_fold)
    prototypes = load_prototypes(arguments.prototypes)
    training_dataset = FeatureDataset(fold.training, arguments.features, prototypes)
    validation_dataset = FeatureDataset(
        fold.validation,
        arguments.features,
        prototypes,
        sample_patches=False,
    )
    loader = DataLoader(
        training_dataset,
        batch_size=int(values["batch_size"]),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_slides,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_slides,
    )
    settings = model_settings(values)
    model = DomainGeneralizableVLM(settings)
    model.initialize_domain_profile(initialize_profile(training_dataset))
    objective = MultiTaskDomainObjective(
        dcr_weight=settings.dcr_weight,
        entropy_weight=settings.entropy_weight,
        dimension=settings.feature_dimension,
        momentum=settings.ema_momentum,
    )
    engine = ExperimentEngine(model, objective, optimizer_settings(values))
    stopping = EarlyStopping(int(values["early_stopping_patience"]))
    runner = ValidationRunner(model, engine.device)
    arguments.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(values["epochs"])):
        engine.train_epoch(loader, epoch)
        predictions = runner.predict(validation_loader)
        evaluation = evaluate_binary(predictions, "idh", None)
        improved = stopping.update(evaluation.metrics.auc_roc)
        if improved:
            engine.save(arguments.output / "best.pt", epoch, seed, stopping.best)
        engine.save(arguments.output / "latest.pt", epoch, seed, stopping.best)
        if stopping.should_stop:
            break


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run(parser().parse_args())


if __name__ == "__main__":
    main()
