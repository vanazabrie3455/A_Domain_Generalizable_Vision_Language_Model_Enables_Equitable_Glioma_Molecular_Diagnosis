from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

ByteImage = NDArray[np.uint8]
FloatImage = NDArray[np.float64]


class SlideReader(Protocol):
    dimensions: tuple[int, int]

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image: ...


@dataclass(frozen=True)
class TileCoordinate:
    x: int
    y: int
    width: int
    height: int
    level: int = 0


@dataclass(frozen=True)
class TileQuality:
    tissue_fraction: float
    background_fraction: float
    laplacian_variance: float
    pen_fraction: float
    accepted: bool


@dataclass(frozen=True)
class TilingSettings:
    patch_size: int = 256
    background_limit: float = 0.7
    laplacian_minimum: float = 50.0
    pen_limit: float = 0.05
    minimum_fragment_area: int = 64
    closing_radius: int = 3


def rgb_array(image: Image.Image) -> ByteImage:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def grayscale(image: ByteImage) -> ByteImage:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def otsu_tissue_mask(image: ByteImage) -> ByteImage:
    gray = grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def remove_small_components(mask: ByteImage, minimum_area: int) -> ByteImage:
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def fill_internal_holes(mask: ByteImage) -> ByteImage:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    buffer = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, buffer, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def morphological_tissue_mask(image: ByteImage, settings: TilingSettings) -> ByteImage:
    mask = otsu_tissue_mask(image)
    radius = settings.closing_radius
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = remove_small_components(closed, settings.minimum_fragment_area)
    return fill_internal_holes(cleaned)


def background_fraction(image: ByteImage) -> float:
    gray = grayscale(image)
    return float(np.mean(gray >= 220))


def laplacian_variance(image: ByteImage) -> float:
    gray = grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def pen_mask(image: ByteImage) -> ByteImage:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, np.array([90, 70, 30]), np.array([140, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 80, 20]), np.array([90, 255, 255]))
    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 45]))
    saturated = cv2.inRange(hsv, np.array([0, 180, 20]), np.array([180, 255, 255]))
    return cv2.bitwise_or(cv2.bitwise_or(blue, green), cv2.bitwise_and(black, saturated))


def pen_fraction(image: ByteImage) -> float:
    return float(np.mean(pen_mask(image) > 0))


def assess_tile(image: ByteImage, settings: TilingSettings) -> TileQuality:
    background = background_fraction(image)
    blur = laplacian_variance(image)
    pen = pen_fraction(image)
    tissue = 1.0 - background
    accepted = (
        background <= settings.background_limit
        and blur >= settings.laplacian_minimum
        and pen <= settings.pen_limit
    )
    return TileQuality(tissue, background, blur, pen, accepted)


def tile_grid(width: int, height: int, patch_size: int = 256) -> Iterator[TileCoordinate]:
    if width < patch_size or height < patch_size:
        return
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            yield TileCoordinate(x, y, patch_size, patch_size)


def tissue_grid(
    mask: ByteImage,
    slide_width: int,
    slide_height: int,
    patch_size: int = 256,
    minimum_tissue: float = 0.3,
) -> Iterator[TileCoordinate]:
    mask_height, mask_width = mask.shape
    scale_x = mask_width / slide_width
    scale_y = mask_height / slide_height
    for coordinate in tile_grid(slide_width, slide_height, patch_size):
        left = int(coordinate.x * scale_x)
        top = int(coordinate.y * scale_y)
        right = max(left + 1, int((coordinate.x + patch_size) * scale_x))
        bottom = max(top + 1, int((coordinate.y + patch_size) * scale_y))
        region = mask[top:bottom, left:right]
        if region.size and float(np.mean(region > 0)) >= minimum_tissue:
            yield coordinate


def read_tile(slide: SlideReader, coordinate: TileCoordinate) -> ByteImage:
    image = slide.read_region(
        (coordinate.x, coordinate.y),
        coordinate.level,
        (coordinate.width, coordinate.height),
    )
    return rgb_array(image)


def accepted_tiles(
    slide: SlideReader,
    coordinates: Iterator[TileCoordinate],
    settings: TilingSettings,
) -> Iterator[tuple[TileCoordinate, ByteImage, TileQuality]]:
    for coordinate in coordinates:
        image = read_tile(slide, coordinate)
        quality = assess_tile(image, settings)
        if quality.accepted:
            yield coordinate, image, quality


def rgb_to_optical_density(image: ByteImage) -> FloatImage:
    values = image.astype(np.float64)
    return -np.log((values + 1.0) / 256.0)


def optical_density_to_rgb(density: FloatImage) -> ByteImage:
    values = 256.0 * np.exp(-density) - 1.0
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def stain_matrix(image: ByteImage, percentile: float = 1.0) -> FloatImage:
    density = rgb_to_optical_density(image).reshape(-1, 3)
    valid = density[np.all(density > 0.15, axis=1)]
    if len(valid) < 3:
        raise ValueError("tile contains insufficient stained pixels")
    covariance = np.cov(valid, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    plane = eigenvectors[:, 1:3]
    projections = valid @ plane
    angles = np.arctan2(projections[:, 1], projections[:, 0])
    low = np.percentile(angles, percentile)
    high = np.percentile(angles, 100.0 - percentile)
    first = plane @ np.array([np.cos(low), np.sin(low)])
    second = plane @ np.array([np.cos(high), np.sin(high)])
    if first[0] < second[0]:
        return np.stack([first, second], axis=1)
    return np.stack([second, first], axis=1)


def stain_concentrations(image: ByteImage, matrix: FloatImage) -> FloatImage:
    density = rgb_to_optical_density(image).reshape(-1, 3).T
    concentrations, _, _, _ = np.linalg.lstsq(matrix, density, rcond=None)
    return concentrations


def macenko_normalize(
    image: ByteImage,
    reference_matrix: FloatImage,
    reference_maximum: FloatImage,
    percentile: float = 1.0,
) -> ByteImage:
    source_matrix = stain_matrix(image, percentile)
    concentrations = stain_concentrations(image, source_matrix)
    source_maximum = np.percentile(concentrations, 99.0, axis=1)
    scaled = concentrations * (reference_maximum / np.maximum(source_maximum, 1e-8))[:, None]
    normalized_density = reference_matrix @ scaled
    normalized = optical_density_to_rgb(normalized_density.T.reshape(image.shape))
    return normalized


def channel_statistics(image: ByteImage) -> tuple[FloatImage, FloatImage]:
    values = image.astype(np.float64).reshape(-1, 3)
    return values.mean(axis=0), values.std(axis=0)


def reinhard_normalize(
    image: ByteImage,
    reference_mean: FloatImage,
    reference_standard_deviation: FloatImage,
) -> ByteImage:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float64)
    source_mean, source_standard_deviation = channel_statistics(lab.astype(np.uint8))
    centered = lab - source_mean
    scaled = centered * (reference_standard_deviation / np.maximum(source_standard_deviation, 1e-8))
    normalized = np.clip(scaled + reference_mean, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)


def save_tile(image: ByteImage, path: Path, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(image).save(temporary, format="JPEG", quality=quality)
    temporary.replace(path)
