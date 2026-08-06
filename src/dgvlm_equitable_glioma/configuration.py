from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

from dgvlm_equitable_glioma.types import ModelSettings, OptimizerSettings

SettingsType = TypeVar("SettingsType", ModelSettings, OptimizerSettings)


def load_configuration(path: Path, overrides: list[str] | None = None) -> DictConfig:
    base = OmegaConf.load(path)
    if overrides:
        override = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, override)
    if not isinstance(base, DictConfig):
        raise TypeError("configuration root must be a mapping")
    OmegaConf.resolve(base)
    return base


def select_values(
    configuration: Mapping[str, Any],
    names: Mapping[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for output_name, configuration_name in names.items():
        if configuration_name not in configuration:
            raise KeyError(f"missing configuration value: {configuration_name}")
        result[output_name] = configuration[configuration_name]
    return result


def dataclass_values(
    configuration: Mapping[str, Any], settings_type: type[SettingsType]
) -> SettingsType:
    names = {field.name for field in fields(settings_type)}
    values = {name: configuration[name] for name in names if name in configuration}
    return settings_type(**values)


def model_settings(configuration: Mapping[str, Any]) -> ModelSettings:
    aliases = {
        "feature_dimension": "feature_dimension",
        "attention_dimension": "attention_dimension",
        "hidden_dimension": "head_hidden_dimension",
        "dropout": "dropout",
        "fsdr_alpha": "fsdr_alpha",
        "text_boost_beta": "text_boost_beta",
        "dcr_weight": "dcr_weight",
        "entropy_weight": "entropy_weight",
        "temperature": "temperature",
        "ema_momentum": "ema_momentum",
    }
    return ModelSettings(**select_values(configuration, aliases))


def optimizer_settings(configuration: Mapping[str, Any]) -> OptimizerSettings:
    aliases = {
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "epochs": "epochs",
        "accumulation": "gradient_accumulation",
        "patience": "early_stopping_patience",
        "restart_period": "restart_period",
        "restart_multiplier": "restart_multiplier",
    }
    return OptimizerSettings(**select_values(configuration, aliases))


def flattened(configuration: DictConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(configuration, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("configuration root must be a mapping")
    return {str(key): value for key, value in container.items()}
