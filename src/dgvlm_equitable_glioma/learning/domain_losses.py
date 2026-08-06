from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional


def covariance(features: Tensor) -> Tensor:
    if features.ndim != 2:
        raise ValueError("features must be a matrix")
    centered = features - features.mean(dim=0, keepdim=True)
    denominator = max(features.shape[0] - 1, 1)
    return centered.transpose(0, 1) @ centered / denominator


def coral_loss(first: Tensor, second: Tensor) -> Tensor:
    if first.shape[1] != second.shape[1]:
        raise ValueError("CORAL feature dimensions must match")
    dimension = first.shape[1]
    mean_difference = torch.mean(torch.square(first.mean(dim=0) - second.mean(dim=0)))
    covariance_difference = torch.mean(torch.square(covariance(first) - covariance(second)))
    return mean_difference + covariance_difference / (4.0 * dimension * dimension)


def multi_domain_coral(features: Mapping[str, Tensor]) -> Tensor:
    domains = sorted(features)
    if len(domains) < 2:
        reference = next(iter(features.values()))
        return reference.sum() * 0.0
    losses = []
    for left in range(len(domains)):
        for right in range(left + 1, len(domains)):
            losses.append(coral_loss(features[domains[left]], features[domains[right]]))
    return torch.stack(losses).mean()


def squared_distance(first: Tensor, second: Tensor) -> Tensor:
    first_norm = torch.sum(first.square(), dim=1, keepdim=True)
    second_norm = torch.sum(second.square(), dim=1, keepdim=True)
    distances = first_norm + second_norm.transpose(0, 1) - 2.0 * first @ second.transpose(0, 1)
    return distances.clamp_min(0.0)


def gaussian_kernel(first: Tensor, second: Tensor, bandwidths: Sequence[float]) -> Tensor:
    distances = squared_distance(first, second)
    kernels = [torch.exp(-distances / (2.0 * bandwidth * bandwidth)) for bandwidth in bandwidths]
    return torch.stack(kernels).mean(dim=0)


def maximum_mean_discrepancy(
    first: Tensor,
    second: Tensor,
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> Tensor:
    within_first = gaussian_kernel(first, first, bandwidths)
    within_second = gaussian_kernel(second, second, bandwidths)
    between = gaussian_kernel(first, second, bandwidths)
    return within_first.mean() + within_second.mean() - 2.0 * between.mean()


def multi_domain_mmd(features: Mapping[str, Tensor]) -> Tensor:
    domains = sorted(features)
    if len(domains) < 2:
        reference = next(iter(features.values()))
        return reference.sum() * 0.0
    values = []
    for left in range(len(domains)):
        for right in range(left + 1, len(domains)):
            values.append(
                maximum_mean_discrepancy(features[domains[left]], features[domains[right]])
            )
    return torch.stack(values).mean()


def mixup(
    features: Tensor,
    labels: Tensor,
    alpha: float = 0.2,
) -> tuple[Tensor, Tensor, Tensor, float]:
    if alpha <= 0.0:
        raise ValueError("mixup alpha must be positive")
    distribution = torch.distributions.Beta(alpha, alpha)
    coefficient = float(distribution.sample().item())
    permutation = torch.randperm(features.shape[0], device=features.device)
    mixed = coefficient * features + (1.0 - coefficient) * features[permutation]
    return mixed, labels, labels[permutation], coefficient


def mixup_cross_entropy(
    logits: Tensor,
    first_labels: Tensor,
    second_labels: Tensor,
    coefficient: float,
) -> Tensor:
    first = functional.cross_entropy(logits, first_labels)
    second = functional.cross_entropy(logits, second_labels)
    return coefficient * first + (1.0 - coefficient) * second


def group_losses(losses: Tensor, groups: Sequence[str]) -> dict[str, Tensor]:
    if losses.ndim != 1 or len(losses) != len(groups):
        raise ValueError("losses and group identifiers must align")
    buckets: dict[str, list[Tensor]] = defaultdict(list)
    for loss, group in zip(losses, groups, strict=True):
        buckets[group].append(loss)
    return {group: torch.stack(values).mean() for group, values in buckets.items()}


class GroupDROWeights:
    def __init__(self, groups: Sequence[str], step_size: float = 0.01) -> None:
        names = sorted(set(groups))
        if not names:
            raise ValueError("GroupDRO requires at least one group")
        self.names = names
        self.step_size = step_size
        self.weights = torch.full((len(names),), 1.0 / len(names))

    def update(self, losses: Mapping[str, Tensor]) -> Tensor:
        values = torch.stack([losses[name] for name in self.names])
        weights = self.weights.to(values)
        weights = weights * torch.exp(self.step_size * values.detach())
        weights = weights / weights.sum()
        self.weights = weights.cpu()
        return torch.sum(weights * values)


def irm_penalty(logits: Tensor, labels: Tensor) -> Tensor:
    scale = torch.ones((), device=logits.device, requires_grad=True)
    loss = functional.cross_entropy(logits * scale, labels)
    gradient = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return gradient.square()


def vrex_penalty(losses: Mapping[str, Tensor]) -> Tensor:
    if len(losses) < 2:
        reference = next(iter(losses.values()))
        return reference.sum() * 0.0
    values = torch.stack(list(losses.values()))
    return values.var(unbiased=False)


def conditional_domain_variance(features: Tensor, labels: Tensor, domains: Tensor) -> Tensor:
    if not (features.shape[0] == labels.shape[0] == domains.shape[0]):
        raise ValueError("features, labels, and domains must align")
    penalties = []
    for label in torch.unique(labels):
        class_features = features[labels == label]
        class_domains = domains[labels == label]
        domain_means = []
        for domain in torch.unique(class_domains):
            selected = class_features[class_domains == domain]
            domain_means.append(selected.mean(dim=0))
        if len(domain_means) >= 2:
            penalties.append(torch.stack(domain_means).var(dim=0, unbiased=False).mean())
    if not penalties:
        return features.sum() * 0.0
    return torch.stack(penalties).mean()
