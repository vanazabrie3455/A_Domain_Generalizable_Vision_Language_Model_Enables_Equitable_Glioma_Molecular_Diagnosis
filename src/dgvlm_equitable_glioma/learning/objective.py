from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from dgvlm_equitable_glioma.types import LossOutput, ModelOutput, TargetName


def attention_entropy(attention: Tensor) -> Tensor:
    probabilities = attention.clamp_min(torch.finfo(attention.dtype).eps)
    return -(probabilities * probabilities.log()).sum()


class DomainMoments(nn.Module):
    def __init__(self, dimension: int = 512, momentum: float = 0.99) -> None:
        super().__init__()
        self.dimension = dimension
        self.momentum = momentum
        self.means: dict[str, Tensor] = {}
        self.variances: dict[str, Tensor] = {}
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, domain: str, representation: Tensor) -> None:
        value = representation.detach()
        if domain not in self.means:
            self.means[domain] = value.clone()
            self.variances[domain] = torch.zeros_like(value)
            self.counts[domain] = 1
            return
        previous_mean = self.means[domain].to(value)
        previous_variance = self.variances[domain].to(value)
        delta = value - previous_mean
        mean = self.momentum * previous_mean + (1.0 - self.momentum) * value
        variance = self.momentum * previous_variance + (1.0 - self.momentum) * delta.square()
        self.means[domain] = mean
        self.variances[domain] = variance
        self.counts[domain] += 1

    def consistency(self, representations: Mapping[str, Tensor]) -> Tensor:
        for domain, representation in representations.items():
            self.update(domain, representation)
        active = sorted(self.means)
        reference = next(iter(representations.values()))
        if len(active) < 2:
            return reference.sum() * 0.0
        means = torch.stack([self.means[domain].to(reference) for domain in active])
        variances = torch.stack([self.variances[domain].to(reference) for domain in active])
        inter = means.var(dim=0, unbiased=False).mean()
        intra = variances.mean()
        return inter / (intra + 1e-8)

    def state(self) -> dict[str, object]:
        return {
            "means": {key: value.cpu() for key, value in self.means.items()},
            "variances": {key: value.cpu() for key, value in self.variances.items()},
            "counts": dict(self.counts),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        raw_means = state["means"]
        raw_variances = state["variances"]
        raw_counts = state["counts"]
        if not isinstance(raw_means, dict) or not isinstance(raw_variances, dict):
            raise TypeError("invalid domain moment state")
        if not isinstance(raw_counts, dict):
            raise TypeError("invalid domain count state")
        self.means = {
            str(key): value for key, value in raw_means.items() if isinstance(value, Tensor)
        }
        self.variances = {
            str(key): value for key, value in raw_variances.items() if isinstance(value, Tensor)
        }
        self.counts = defaultdict(int, {str(key): int(value) for key, value in raw_counts.items()})


class MultiTaskDomainObjective(nn.Module):
    def __init__(
        self,
        class_weights: Mapping[TargetName, Tensor] | None = None,
        task_weights: Mapping[TargetName, float] | None = None,
        dcr_weight: float = 0.1,
        entropy_weight: float = 0.001,
        dimension: int = 512,
        momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.class_weights = dict(class_weights or {})
        self.task_weights = dict(task_weights or {})
        self.dcr_weight = dcr_weight
        self.entropy_weight = entropy_weight
        self.moments = DomainMoments(dimension, momentum)

    def task_loss(
        self,
        outputs: Sequence[ModelOutput],
        labels: Sequence[Mapping[TargetName, int]],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        reference = outputs[0].representation
        total = reference.sum() * 0.0
        components: dict[str, Tensor] = {}
        for target in ("idh", "codeletion", "mgmt", "subtype"):
            selected = [index for index, record in enumerate(labels) if target in record]
            if not selected:
                continue
            logits = torch.cat([outputs[index].logits[target] for index in selected], dim=0)
            targets = torch.tensor(
                [labels[index][target] for index in selected],
                dtype=torch.long,
                device=logits.device,
            )
            class_weight = self.class_weights.get(target)
            if class_weight is not None:
                class_weight = class_weight.to(logits)
            loss = functional.cross_entropy(logits, targets, weight=class_weight)
            weight = self.task_weights.get(target, 1.0)
            components[target] = loss
            total = total + weight * loss
        return total, components

    def forward(
        self,
        outputs: Sequence[ModelOutput],
        labels: Sequence[Mapping[TargetName, int]],
        domains: Sequence[str],
    ) -> LossOutput:
        task, components = self.task_loss(outputs, labels)
        grouped: dict[str, list[Tensor]] = defaultdict(list)
        for output, domain in zip(outputs, domains, strict=True):
            grouped[domain].append(output.representation)
        representatives = {
            domain: torch.stack(values).mean(dim=0) for domain, values in grouped.items()
        }
        consistency = self.moments.consistency(representatives)
        entropy = torch.stack([attention_entropy(output.attention) for output in outputs]).mean()
        total = task + self.dcr_weight * consistency + self.entropy_weight * entropy
        return LossOutput(total, task, consistency, entropy, components)


def inverse_frequency_weights(labels: Sequence[int], classes: int) -> Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=classes).float()
    if bool((counts == 0).any()):
        raise ValueError("each class must occur when computing class weights")
    inverse = counts.sum() / counts
    return inverse / inverse.mean()
