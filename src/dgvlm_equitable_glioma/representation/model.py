from collections.abc import Mapping

from torch import Tensor, nn

from dgvlm_equitable_glioma.representation.attention import DomainAwareAttention
from dgvlm_equitable_glioma.representation.heads import MolecularHeads
from dgvlm_equitable_glioma.representation.randomization import FeatureDomainRandomizer
from dgvlm_equitable_glioma.types import ModelOutput, ModelSettings, TargetName


class DomainGeneralizableVLM(nn.Module):
    def __init__(
        self,
        settings: ModelSettings | None = None,
        classes: Mapping[TargetName, int] | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings or ModelSettings()
        self.randomizer = FeatureDomainRandomizer(
            self.settings.feature_dimension,
            self.settings.fsdr_alpha,
        )
        self.aggregator = DomainAwareAttention(
            self.settings.feature_dimension,
            self.settings.attention_dimension,
            self.settings.text_boost_beta,
            self.settings.temperature,
        )
        self.heads = MolecularHeads(
            self.settings.feature_dimension,
            self.settings.hidden_dimension,
            self.settings.dropout,
            classes,
        )

    def forward(self, features: Tensor, prototypes: Tensor) -> ModelOutput:
        randomized = self.randomizer(features)
        representation, attention, scores = self.aggregator(randomized, prototypes)
        logits = self.heads(representation)
        return ModelOutput(logits, representation, attention, scores)

    def initialize_domain_profile(self, profile: Tensor) -> None:
        self.randomizer.statistics.set_profile(profile)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
