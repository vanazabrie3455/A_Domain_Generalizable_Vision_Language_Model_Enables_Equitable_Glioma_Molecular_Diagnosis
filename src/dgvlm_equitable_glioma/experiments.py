from dataclasses import dataclass, replace
from typing import Literal

Backbone = Literal["conch", "uni", "virchow", "ctranspath", "resnet50"]
Pooling = Literal["gated", "mean", "max", "transformer", "clustering"]
Normalization = Literal["macenko", "reinhard", "raw"]


@dataclass(frozen=True)
class Experiment:
    name: str
    backbone: Backbone = "conch"
    pooling: Pooling = "gated"
    normalization: Normalization = "macenko"
    fsdr_alpha: float = 0.3
    text_boost_beta: float = 0.5
    dcr_weight: float = 0.1
    entropy_weight: float = 0.001
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    training_fraction: float = 1.0
    standard_augmentation: bool = True
    domain_adversarial: bool = False
    coral: bool = False
    mixup: bool = False
    group_dro: bool = False
    irm: bool = False
    vr_ex: bool = False


def primary() -> Experiment:
    return Experiment(name="dgvlm")


def component_ablations() -> tuple[Experiment, ...]:
    full = primary()
    return (
        full,
        replace(full, name="without_fsdr", fsdr_alpha=0.0),
        replace(full, name="without_text_modulation", text_boost_beta=0.0),
        replace(full, name="without_dcr", dcr_weight=0.0),
        replace(
            full,
            name="conch_clam_no_dg",
            fsdr_alpha=0.0,
            text_boost_beta=0.0,
            dcr_weight=0.0,
        ),
    )


def backbone_ablations() -> tuple[Experiment, ...]:
    full = primary()
    return (
        full,
        replace(full, name="uni_dg", backbone="uni"),
        replace(full, name="virchow_dg", backbone="virchow"),
        replace(full, name="ctranspath_dg", backbone="ctranspath"),
        replace(full, name="resnet50_dg", backbone="resnet50"),
    )


def pooling_ablations() -> tuple[Experiment, ...]:
    full = primary()
    return (
        full,
        replace(full, name="mean_pooling", pooling="mean"),
        replace(full, name="max_pooling", pooling="max"),
        replace(full, name="transformer_pooling", pooling="transformer"),
        replace(full, name="clustering_pooling", pooling="clustering"),
    )


def fsdr_sensitivity() -> tuple[Experiment, ...]:
    full = primary()
    return tuple(
        replace(full, name=f"fsdr_alpha_{value:g}", fsdr_alpha=value)
        for value in (0.0, 0.1, 0.3, 0.5, 1.0)
    )


def text_sensitivity() -> tuple[Experiment, ...]:
    full = primary()
    return tuple(
        replace(full, name=f"text_beta_{value:g}", text_boost_beta=value)
        for value in (0.0, 0.2, 0.5, 1.0, 2.0)
    )


def dcr_sensitivity() -> tuple[Experiment, ...]:
    full = primary()
    return tuple(
        replace(full, name=f"dcr_lambda_{value:g}", dcr_weight=value)
        for value in (0.0, 0.01, 0.1, 0.5, 1.0)
    )


def learning_rate_sensitivity() -> tuple[Experiment, ...]:
    full = primary()
    return tuple(
        replace(full, name=f"learning_rate_{value:g}", learning_rate=value)
        for value in (0.00005, 0.0001, 0.0005)
    )


def stain_ablations() -> tuple[Experiment, ...]:
    full = primary()
    return (
        full,
        replace(full, name="reinhard", normalization="reinhard"),
        replace(full, name="raw_he", normalization="raw"),
    )


def augmentation_ablations() -> tuple[Experiment, ...]:
    full = primary()
    return (
        replace(full, name="no_augmentation", fsdr_alpha=0.0, standard_augmentation=False),
        replace(full, name="standard_only", fsdr_alpha=0.0, standard_augmentation=True),
        replace(full, name="fsdr_only", standard_augmentation=False),
        full,
    )


def training_scale_ablations() -> tuple[Experiment, ...]:
    full = primary()
    return tuple(
        replace(full, name=f"training_fraction_{value:g}", training_fraction=value)
        for value in (0.1, 0.25, 0.5, 0.75, 1.0)
    )


def domain_generalization_baselines() -> tuple[Experiment, ...]:
    base = replace(
        primary(),
        fsdr_alpha=0.0,
        text_boost_beta=0.0,
        dcr_weight=0.0,
    )
    return (
        replace(base, name="erm"),
        replace(base, name="domain_adversarial", domain_adversarial=True),
        replace(base, name="coral", coral=True),
        replace(base, name="mixup", mixup=True),
        replace(base, name="group_dro", group_dro=True),
        replace(base, name="irm", irm=True),
        replace(base, name="vrex", vr_ex=True),
    )


def full_registry() -> dict[str, Experiment]:
    groups = (
        component_ablations(),
        backbone_ablations(),
        pooling_ablations(),
        fsdr_sensitivity(),
        text_sensitivity(),
        dcr_sensitivity(),
        learning_rate_sensitivity(),
        stain_ablations(),
        augmentation_ablations(),
        training_scale_ablations(),
        domain_generalization_baselines(),
    )
    registry: dict[str, Experiment] = {}
    for group in groups:
        for experiment in group:
            existing = registry.get(experiment.name)
            if existing is not None and existing != experiment:
                raise ValueError(f"experiment name collision: {experiment.name}")
            registry[experiment.name] = experiment
    return registry


def validate_experiment(experiment: Experiment) -> None:
    if experiment.fsdr_alpha < 0.0:
        raise ValueError("FSDR alpha cannot be negative")
    if experiment.text_boost_beta < 0.0:
        raise ValueError("text boost cannot be negative")
    if experiment.dcr_weight < 0.0:
        raise ValueError("DCR weight cannot be negative")
    if experiment.entropy_weight < 0.0:
        raise ValueError("entropy weight cannot be negative")
    if experiment.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if experiment.weight_decay < 0.0:
        raise ValueError("weight decay cannot be negative")
    if not 0.0 < experiment.training_fraction <= 1.0:
        raise ValueError("training fraction must be in (0, 1]")
    if experiment.backbone != "conch" and experiment.text_boost_beta > 0.0:
        raise ValueError("text modulation requires aligned prototype embeddings for the backbone")


def validate_registry(registry: dict[str, Experiment]) -> None:
    if not registry:
        raise ValueError("experiment registry cannot be empty")
    for name, experiment in registry.items():
        if name != experiment.name:
            raise ValueError("registry key must equal experiment name")
        validate_experiment(experiment)
