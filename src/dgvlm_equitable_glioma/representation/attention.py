import torch
from torch import Tensor, nn
from torch.nn import functional


class TextAlignment(nn.Module):
    def __init__(self, initial_temperature: float = 0.07) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp_min(1e-4)

    def forward(self, features: Tensor, prototypes: Tensor) -> Tensor:
        normalized_features = functional.normalize(features, dim=-1)
        normalized_prototypes = functional.normalize(prototypes, dim=-1)
        similarities = normalized_features @ normalized_prototypes.transpose(-1, -2)
        return similarities.max(dim=-1).values / self.temperature


class GatedAttention(nn.Module):
    def __init__(self, feature_dimension: int = 512, attention_dimension: int = 256) -> None:
        super().__init__()
        self.tanh_projection = nn.Linear(feature_dimension, attention_dimension)
        self.gate_projection = nn.Linear(feature_dimension, attention_dimension)
        self.score_projection = nn.Linear(attention_dimension, 1, bias=False)

    def logits(self, features: Tensor) -> Tensor:
        content = torch.tanh(self.tanh_projection(features))
        gate = torch.sigmoid(self.gate_projection(features))
        return self.score_projection(content * gate).squeeze(-1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        attention = torch.softmax(self.logits(features), dim=0)
        representation = torch.sum(attention.unsqueeze(-1) * features, dim=0)
        return representation, attention


class DomainAwareAttention(nn.Module):
    def __init__(
        self,
        feature_dimension: int = 512,
        attention_dimension: int = 256,
        beta: float = 0.5,
        initial_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.base = GatedAttention(feature_dimension, attention_dimension)
        self.alignment = TextAlignment(initial_temperature)

    def forward(self, features: Tensor, prototypes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        text_scores = self.alignment(features, prototypes)
        modulation = 1.0 + self.beta * torch.sigmoid(text_scores)
        attention_logits = self.base.logits(features) * modulation
        attention = torch.softmax(attention_logits, dim=0)
        representation = torch.sum(attention.unsqueeze(-1) * features, dim=0)
        return representation, attention, text_scores
