import argparse
import csv
import logging
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dgvlm_equitable_glioma.cohorts.features import FeatureDataset, collate_slides
from dgvlm_equitable_glioma.cohorts.manifest import load_manifest
from dgvlm_equitable_glioma.configuration import flattened, load_configuration, model_settings
from dgvlm_equitable_glioma.learning.validation import ValidationRunner, grouped_evaluations
from dgvlm_equitable_glioma.representation.model import DomainGeneralizableVLM


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="dgvlm-evaluate")
    result.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--features", type=Path, required=True)
    result.add_argument("--prototypes", type=Path, required=True)
    result.add_argument("--weights", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("override", nargs="*")
    return result


def run(arguments: argparse.Namespace) -> None:
    values = flattened(load_configuration(arguments.config, arguments.override))
    prototypes = torch.load(arguments.prototypes, map_location="cpu")
    if not isinstance(prototypes, torch.Tensor):
        raise TypeError("prototype file must contain a tensor")
    manifest = load_manifest(arguments.manifest)
    dataset = FeatureDataset(manifest, arguments.features, prototypes, sample_patches=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_slides,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomainGeneralizableVLM(model_settings(values))
    payload = torch.load(arguments.weights, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.to(device)
    predictions = ValidationRunner(model, device).predict(loader)
    evaluations = grouped_evaluations(predictions)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["domain", "target", "samples", *asdict(evaluations[0].metrics).keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for evaluation in evaluations:
            row = {
                "domain": evaluation.domain,
                "target": evaluation.target,
                "samples": evaluation.samples,
                **asdict(evaluation.metrics),
            }
            writer.writerow(row)
    temporary.replace(arguments.output)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(parser().parse_args())


if __name__ == "__main__":
    main()
