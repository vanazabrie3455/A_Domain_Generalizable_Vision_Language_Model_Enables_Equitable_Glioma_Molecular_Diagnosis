import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dgvlm_equitable_glioma.learning.checkpoint import save_checkpoint
from dgvlm_equitable_glioma.learning.objective import MultiTaskDomainObjective
from dgvlm_equitable_glioma.types import FeatureRecord, LossOutput, ModelOutput, OptimizerSettings

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    loss: float
    task_loss: float
    consistency_loss: float
    entropy: float
    learning_rate: float
    optimizer_steps: int


class GradientAccumulator:
    def __init__(self, optimizer: Optimizer, steps: int) -> None:
        if steps < 1:
            raise ValueError("accumulation steps must be positive")
        self.optimizer = optimizer
        self.steps = steps
        self.pending = 0

    def backward(
        self,
        loss: Tensor,
        scaler: torch.cuda.amp.GradScaler,
        parameters: Iterable[nn.Parameter],
        clip_norm: float | None = None,
    ) -> bool:
        scaler.scale(loss / self.steps).backward()
        self.pending += 1
        if self.pending < self.steps:
            return False
        if clip_norm is not None:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = 0
        return True

    def flush(
        self,
        scaler: torch.cuda.amp.GradScaler,
        parameters: Iterable[nn.Parameter],
        clip_norm: float | None = None,
    ) -> bool:
        if self.pending == 0:
            return False
        correction = self.steps / self.pending
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        if clip_norm is not None:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = 0
        return True


class EarlyStopping:
    def __init__(self, patience: int = 20, minimum_delta: float = 0.0) -> None:
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.best = float("-inf")
        self.wait = 0

    def update(self, metric: float) -> bool:
        if metric > self.best + self.minimum_delta:
            self.best = metric
            self.wait = 0
            return True
        self.wait += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.wait >= self.patience


class ExperimentEngine:
    def __init__(
        self,
        model: nn.Module,
        objective: MultiTaskDomainObjective,
        settings: OptimizerSettings | None = None,
        device: torch.device | None = None,
        mixed_precision: bool = True,
    ) -> None:
        self.model = model
        self.objective = objective
        self.settings = settings or OptimizerSettings()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.settings.restart_period,
            T_mult=self.settings.restart_multiplier,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.accumulator = GradientAccumulator(self.optimizer, self.settings.accumulation)
        self.global_step = 0

    def _forward_record(self, record: FeatureRecord) -> ModelOutput:
        features = record.features.to(self.device, non_blocking=True)
        prototypes = record.text_embeddings.to(self.device, non_blocking=True)
        return self.model(features, prototypes)

    def train_batch(self, records: Sequence[FeatureRecord]) -> LossOutput:
        self.model.train()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.mixed_precision,
        ):
            outputs = [self._forward_record(record) for record in records]
            result = self.objective(
                outputs,
                [record.labels for record in records],
                [record.domain for record in records],
            )
        stepped = self.accumulator.backward(result.total, self.scaler, self.model.parameters())
        if stepped:
            self.global_step += 1
        return result

    def train_epoch(self, batches: Iterable[Sequence[FeatureRecord]], epoch: int) -> EpochSummary:
        totals: list[float] = []
        tasks: list[float] = []
        consistencies: list[float] = []
        entropies: list[float] = []
        start_steps = self.global_step
        for records in batches:
            result = self.train_batch(records)
            totals.append(float(result.total.detach().cpu()))
            tasks.append(float(result.task.detach().cpu()))
            consistencies.append(float(result.consistency.detach().cpu()))
            entropies.append(float(result.entropy.detach().cpu()))
        if self.accumulator.flush(self.scaler, self.model.parameters()):
            self.global_step += 1
        self.scheduler.step(epoch + 1)
        summary = EpochSummary(
            epoch=epoch,
            loss=float(np.mean(totals)),
            task_loss=float(np.mean(tasks)),
            consistency_loss=float(np.mean(consistencies)),
            entropy=float(np.mean(entropies)),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            optimizer_steps=self.global_step - start_steps,
        )
        LOGGER.info("epoch=%d loss=%.6f task=%.6f", epoch, summary.loss, summary.task_loss)
        return summary

    def save(self, path: Path, epoch: int, seed: int, best_metric: float) -> None:
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            epoch,
            self.global_step,
            seed,
            best_metric,
            {"domain_moments": self.objective.moments.state()},
        )
