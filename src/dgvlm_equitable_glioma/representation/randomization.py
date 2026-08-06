from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import Tensor, nn


class InterDomainProfile(nn.Module):
    def __init__(self, dimension: int = 512) -> None:
        super().__init__()
        self.dimension = dimension
        self.register_buffer("profile", torch.zeros(dimension))
        self.register_buffer("initialized", torch.tensor(False))

    def estimate(self, domain_features: Iterable[tuple[str, Tensor]]) -> Tensor:
        sums: dict[str, Tensor] = {}
        counts: dict[str, int] = defaultdict(int)
        for domain, features in domain_features:
            value = features.detach().float().mean(dim=0)
            if domain not in sums:
                sums[domain] = torch.zeros_like(value)
            sums[domain] += value
            counts[domain] += 1
        if len(sums) < 2:
            raise ValueError("at least two source domains are required")
        means = torch.stack([sums[name] / counts[name] for name in sorted(sums)])
        estimate = means.std(dim=0, unbiased=True)
        self.profile.copy_(estimate.to(self.profile.device))
        self.initialized.fill_(True)
        return self.profile

    def set_profile(self, profile: Tensor) -> None:
        if profile.shape != (self.dimension,):
            raise ValueError(
                f"expected profile shape {(self.dimension,)}, received {profile.shape}"
            )
        self.profile.copy_(profile.detach().to(self.profile))
        self.initialized.fill_(True)


class FeatureDomainRandomizer(nn.Module):
    def __init__(self, dimension: int = 512, alpha: float = 0.3) -> None:
        super().__init__()
        self.alpha = alpha
        self.statistics = InterDomainProfile(dimension)

    def forward(self, features: Tensor) -> Tensor:
        if not self.training or self.alpha == 0.0:
            return features
        if not bool(self.statistics.initialized.item()):
            raise RuntimeError("inter-domain profile has not been initialized")
        scale = self.alpha * self.statistics.profile.to(features)
        noise = torch.randn_like(features) * scale
        return features + noise
