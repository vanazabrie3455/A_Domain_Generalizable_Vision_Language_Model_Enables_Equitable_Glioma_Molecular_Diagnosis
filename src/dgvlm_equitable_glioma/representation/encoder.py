from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from dgvlm_equitable_glioma.cohorts.tiles import ByteImage


class ImageEncoder(Protocol):
    def encode_image(self, image: Tensor) -> Tensor: ...


class TextEncoder(Protocol):
    def encode_text(self, text: list[str]) -> Tensor: ...


GLIOMA_PROTOTYPES = (
    "high cellularity with nuclear pleomorphism",
    "pseudopalisading necrosis",
    "microvascular proliferation",
    "oligodendroglial morphology with perinuclear halos",
    "gemistocytic astrocytic features",
    "necrotic tissue",
    "hemorrhage",
    "normal brain parenchyma",
    "reactive gliosis",
)


class TileFileDataset(Dataset[tuple[str, Tensor]]):
    def __init__(self, paths: Iterable[Path], transform: Callable[[Image.Image], Tensor]) -> None:
        self.paths = tuple(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return path.stem, tensor


def conch_transform(size: int = 224) -> Callable[[Image.Image], Tensor]:
    return Compose(
        [
            Resize((size, size), antialias=True),
            ToTensor(),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class FrozenImageEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, output_dimension: int = 512) -> None:
        super().__init__()
        self.encoder = encoder
        self.output_dimension = output_dimension
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "FrozenImageEncoder":
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        features = self.encoder(images)
        if isinstance(features, tuple):
            features = features[0]
        if not isinstance(features, Tensor):
            raise TypeError("image encoder output must be a tensor")
        if features.ndim != 2 or features.shape[-1] != self.output_dimension:
            raise ValueError(f"unexpected image feature shape: {features.shape}")
        return features


def encode_tiles(
    encoder: FrozenImageEncoder,
    paths: Iterable[Path],
    device: torch.device,
    batch_size: int = 256,
    workers: int = 8,
) -> tuple[list[str], Tensor]:
    dataset = TileFileDataset(paths, conch_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    encoder.to(device)
    identifiers: list[str] = []
    batches: list[Tensor] = []
    for names, images in loader:
        features = encoder(images.to(device, non_blocking=True))
        identifiers.extend(names)
        batches.append(features.cpu())
    if not batches:
        return identifiers, torch.empty((0, encoder.output_dimension))
    return identifiers, torch.cat(batches, dim=0)


def write_features(path: Path, identifiers: list[str], features: Tensor) -> None:
    if features.shape != (len(identifiers), 512):
        raise ValueError("feature rows must align with tile identifiers")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = np.asarray(identifiers, dtype=h5py.string_dtype(encoding="utf-8"))
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("features", data=features.float().numpy(), compression="gzip")
        handle.create_dataset("tile_ids", data=encoded, compression="gzip")
    temporary.replace(path)


def read_rgb(path: Path) -> ByteImage:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def encode_text_prototypes(
    encoder: TextEncoder,
    prompts: tuple[str, ...] = GLIOMA_PROTOTYPES,
) -> Tensor:
    with torch.no_grad():
        embeddings = encoder.encode_text(list(prompts))
    if embeddings.shape != (len(prompts), 512):
        raise ValueError(f"unexpected text embedding shape: {embeddings.shape}")
    return torch.nn.functional.normalize(embeddings.float(), dim=-1)


def save_prototypes(path: Path, prototypes: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(prototypes.cpu(), temporary)
    temporary.replace(path)
