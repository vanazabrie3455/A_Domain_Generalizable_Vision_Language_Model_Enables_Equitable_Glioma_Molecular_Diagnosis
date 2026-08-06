import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from dgvlm_equitable_glioma.representation.heads import MolecularHeads
from dgvlm_equitable_glioma.types import ModelOutput, TargetName


def empty_scores(features: Tensor) -> Tensor:
    return torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)


class MeanMIL(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        hidden_dimension: int = 256,
        dropout: float = 0.3,
        classes: Mapping[TargetName, int] | None = None,
    ) -> None:
        super().__init__()
        self.heads = MolecularHeads(dimension, hidden_dimension, dropout, classes)

    def forward(self, features: Tensor, prototypes: Tensor) -> ModelOutput:
        del prototypes
        representation = features.mean(dim=0)
        attention = torch.full(
            (features.shape[0],),
            1.0 / features.shape[0],
            device=features.device,
            dtype=features.dtype,
        )
        return ModelOutput(
            self.heads(representation),
            representation,
            attention,
            empty_scores(features),
        )


class MaxMIL(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        hidden_dimension: int = 256,
        dropout: float = 0.3,
        classes: Mapping[TargetName, int] | None = None,
    ) -> None:
        super().__init__()
        self.instance_score = nn.Linear(dimension, 1)
        self.heads = MolecularHeads(dimension, hidden_dimension, dropout, classes)

    def forward(self, features: Tensor, prototypes: Tensor) -> ModelOutput:
        del prototypes
        scores = self.instance_score(features).squeeze(-1)
        index = scores.argmax()
        representation = features[index]
        attention = torch.zeros_like(scores)
        attention[index] = 1.0
        return ModelOutput(
            self.heads(representation),
            representation,
            attention,
            empty_scores(features),
        )


class StandardAttentionMIL(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        attention_dimension: int = 256,
        hidden_dimension: int = 256,
        dropout: float = 0.3,
        classes: Mapping[TargetName, int] | None = None,
    ) -> None:
        super().__init__()
        self.content = nn.Linear(dimension, attention_dimension)
        self.gate = nn.Linear(dimension, attention_dimension)
        self.score = nn.Linear(attention_dimension, 1, bias=False)
        self.heads = MolecularHeads(dimension, hidden_dimension, dropout, classes)

    def forward(self, features: Tensor, prototypes: Tensor) -> ModelOutput:
        del prototypes
        hidden = torch.tanh(self.content(features)) * torch.sigmoid(self.gate(features))
        scores = self.score(hidden).squeeze(-1)
        attention = torch.softmax(scores, dim=0)
        representation = torch.sum(attention.unsqueeze(-1) * features, dim=0)
        return ModelOutput(
            self.heads(representation),
            representation,
            attention,
            empty_scores(features),
        )


class ClusteringAttentionMIL(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        attention_dimension: int = 256,
        hidden_dimension: int = 256,
        dropout: float = 0.3,
        clusters: int = 8,
        classes: Mapping[TargetName, int] | None = None,
    ) -> None:
        super().__init__()
        self.clusters = clusters
        self.assignment = nn.Sequential(
            nn.Linear(dimension, attention_dimension),
            nn.Tanh(),
            nn.Linear(attention_dimension, clusters),
        )
        self.cluster_attention = nn.Linear(dimension, 1)
        self.heads = MolecularHeads(dimension, hidden_dimension, dropout, classes)

    def forward(self, features: Tensor, prototypes: Tensor) -> ModelOutput:
        del prototypes
        assignments = torch.softmax(self.assignment(features), dim=-1)
        cluster_mass = assignments.sum(dim=0).clamp_min(1e-8)
        cluster_features = assignments.transpose(0, 1) @ features
        cluster_features = cluster_features / cluster_mass.unsqueeze(-1)
        cluster_scores = self.cluster_attention(cluster_features).squeeze(-1)
        cluster_weights = torch.softmax(cluster_scores, dim=0)
        representation = torch.sum(cluster_weights.unsqueeze(-1) * cluster_features, dim=0)
        patch_attention = assignments @ cluster_weights
        patch_attention = patch_attention / patch_attention.sum()
        return ModelOutput(
            self.heads(representation),
            representation,
            patch_attention,
            empty_scores(features),
        )


class NyströmAttention(nn.Module):
    def __init__(self, dimension: int = 512, heads: int = 8, landmarks: int = 64) -> None:
        super().__init__()
        if dimension % heads != 0:
            raise ValueError("feature dimension must be divisible by attention heads")
        self.dimension = dimension
        self.heads = heads
        self.landmarks = landmarks
        self.head_dimension = dimension // heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)

    def _heads(self, values: Tensor) -> Tensor:
        return values.reshape(values.shape[0], self.heads, self.head_dimension).transpose(0, 1)

    def _landmark_mean(self, values: Tensor) -> Tensor:
        sequence = values.shape[1]
        groups = min(self.landmarks, sequence)
        padding = (groups - sequence % groups) % groups
        if padding:
            tail = values[:, -1:].expand(-1, padding, -1)
            values = torch.cat([values, tail], dim=1)
        group_size = values.shape[1] // groups
        return values.reshape(self.heads, groups, group_size, self.head_dimension).mean(dim=2)

    def forward(self, features: Tensor) -> Tensor:
        queries = self._heads(self.query(features))
        keys = self._heads(self.key(features))
        values = self._heads(self.value(features))
        landmark_queries = self._landmark_mean(queries)
        landmark_keys = self._landmark_mean(keys)
        scale = 1.0 / math.sqrt(self.head_dimension)
        first = torch.softmax(queries @ landmark_keys.transpose(-1, -2) * scale, dim=-1)
        middle = torch.softmax(landmark_queries @ landmark_keys.transpose(-1, -2) * scale, dim=-1)
        last = torch.softmax(landmark_queries @ keys.transpose(-1, -2) * scale, dim=-1)
        inverse = torch.linalg.pinv(middle.float()).to(middle)
        attended = first @ inverse @ last @ values
        merged = attended.transpose(0, 1).reshape(features.shape[0], self.dimension)
        return self.output(merged)


class TransformerMIL(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        heads: int = 8,
        landmarks: int = 64,
        hidden_dimension: int = 256,
        dropout: float = 0.3,
        classes: Mapping[TargetName, int] | None = None,
    ) -> None:
        super().__init__()
        self.class_token = nn.Parameter(torch.zeros(1, dimension))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.first_norm = nn.LayerNorm(dimension)
        self.attention = NyströmAttention(dimension, heads, landmarks)
        self.second_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
        )
        self.heads = MolecularHeads(dimension, hidden_dimension, dropout, classes)

    def forward(self, features: Tensor, prototypes: Tensor) -> ModelOutput:
        del prototypes
        sequence = torch.cat([self.class_token, features], dim=0)
        sequence = sequence + self.attention(self.first_norm(sequence))
        sequence = sequence + self.feed_forward(self.second_norm(sequence))
        representation = sequence[0]
        patch_scores = torch.cosine_similarity(features, representation.unsqueeze(0), dim=-1)
        attention = torch.softmax(patch_scores, dim=0)
        return ModelOutput(
            self.heads(representation),
            representation,
            attention,
            empty_scores(features),
        )


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, features: Tensor, weight: float) -> Tensor:
        ctx.weight = weight
        return features.view_as(features)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        weight = float(ctx.weight)
        return -weight * gradient, None


class DomainAdversary(nn.Module):
    def __init__(self, dimension: int, domains: int, hidden_dimension: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, domains),
        )

    def forward(self, features: Tensor, weight: float = 1.0) -> Tensor:
        reversed_features = GradientReversalFunction.apply(features, weight)
        return self.network(reversed_features)
