from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from dgvlm_equitable_glioma.cohorts.manifest import Manifest
from dgvlm_equitable_glioma.types import TargetName


@dataclass(frozen=True)
class DomainFold:
    held_out: str
    training: Manifest
    validation: Manifest
    testing: Manifest


class LeaveOneDomainOut:
    def __init__(
        self,
        manifest: Manifest,
        validation_folds: int = 5,
        seed: int = 42,
        stratify_target: TargetName = "idh",
    ) -> None:
        self.manifest = manifest
        self.validation_folds = validation_folds
        self.seed = seed
        self.stratify_target = stratify_target

    def split(self, held_out: str, validation_fold: int = 0) -> DomainFold:
        testing = self.manifest.select_domains([held_out])
        source = self.manifest.exclude_domains([held_out])
        eligible = tuple(
            record for record in source.records if self.stratify_target in record.available
        )
        labels = np.asarray([record.labels[self.stratify_target] for record in eligible])
        splitter = StratifiedKFold(
            n_splits=self.validation_folds,
            shuffle=True,
            random_state=self.seed,
        )
        choices = list(splitter.split(np.zeros(len(eligible)), labels))
        train_indices, validation_indices = choices[validation_fold]
        training_records = tuple(eligible[index] for index in train_indices)
        validation_records = tuple(eligible[index] for index in validation_indices)
        from dgvlm_equitable_glioma.cohorts.manifest import _records_digest

        training = Manifest(training_records, _records_digest(training_records))
        validation = Manifest(validation_records, _records_digest(validation_records))
        return DomainFold(held_out, training, validation, testing)

    def __iter__(self) -> Iterator[DomainFold]:
        for domain in self.manifest.domains:
            yield self.split(domain)
