from collections.abc import Mapping

from torch import Tensor, nn
from torch.nn import functional

from dgvlm_equitable_glioma.types import TargetName


class SingleSlideBatchNorm(nn.BatchNorm1d):
    def forward(self, features: Tensor) -> Tensor:
        use_batch_statistics = self.training and features.shape[0] > 1
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        return functional.batch_norm(
            features,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            use_batch_statistics,
            exponential_average_factor,
            self.eps,
        )


class MolecularHead(nn.Module):
    def __init__(
        self, input_dimension: int, hidden_dimension: int, classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            SingleSlideBatchNorm(hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, classes),
        )

    def forward(self, representation: Tensor) -> Tensor:
        if representation.ndim == 1:
            representation = representation.unsqueeze(0)
        return self.layers(representation)


class MolecularHeads(nn.Module):
    def __init__(
        self,
        input_dimension: int = 512,
        hidden_dimension: int = 256,
        dropout: float = 0.3,
        classes: Mapping[TargetName, int] | None = None,
    ) -> None:
        super().__init__()
        specification = classes or {"idh": 2, "codeletion": 2, "mgmt": 2, "subtype": 3}
        self.heads = nn.ModuleDict(
            {
                target: MolecularHead(input_dimension, hidden_dimension, count, dropout)
                for target, count in specification.items()
            }
        )

    def forward(self, representation: Tensor) -> dict[TargetName, Tensor]:
        return {target: head(representation) for target, head in self.heads.items()}
