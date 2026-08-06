import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from dgvlm_equitable_glioma.cohorts.tiles import ByteImage

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AugmentationSettings:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotation_probability: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05


def horizontal_flip(image: ByteImage) -> ByteImage:
    return np.ascontiguousarray(image[:, ::-1])


def vertical_flip(image: ByteImage) -> ByteImage:
    return np.ascontiguousarray(image[::-1, :])


def rotate_quarter(image: ByteImage, turns: int) -> ByteImage:
    return np.ascontiguousarray(np.rot90(image, turns % 4))


def brightness_shift(image: ByteImage, factor: float) -> ByteImage:
    values = image.astype(np.float32) * factor
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def contrast_shift(image: ByteImage, factor: float) -> ByteImage:
    values = image.astype(np.float32)
    means = values.mean(axis=(0, 1), keepdims=True)
    shifted = (values - means) * factor + means
    return np.clip(shifted, 0.0, 255.0).astype(np.uint8)


def saturation_shift(image: ByteImage, factor: float) -> ByteImage:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def hue_shift(image: ByteImage, fraction: float) -> ByteImage:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
    offset = int(round(fraction * 180.0))
    hsv[..., 0] = (hsv[..., 0] + offset) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class StandardAugmentation:
    def __init__(self, settings: AugmentationSettings | None = None, seed: int = 42) -> None:
        self.settings = settings or AugmentationSettings()
        self.generator = np.random.default_rng(seed)

    def uniform_factor(self, magnitude: float) -> float:
        return float(self.generator.uniform(1.0 - magnitude, 1.0 + magnitude))

    def __call__(self, image: ByteImage) -> ByteImage:
        result = image
        if self.generator.random() < self.settings.horizontal_flip_probability:
            result = horizontal_flip(result)
        if self.generator.random() < self.settings.vertical_flip_probability:
            result = vertical_flip(result)
        if self.generator.random() < self.settings.rotation_probability:
            result = rotate_quarter(result, int(self.generator.integers(0, 4)))
        result = brightness_shift(result, self.uniform_factor(self.settings.brightness))
        result = contrast_shift(result, self.uniform_factor(self.settings.contrast))
        result = saturation_shift(result, self.uniform_factor(self.settings.saturation))
        result = hue_shift(
            result,
            float(self.generator.uniform(-self.settings.hue, self.settings.hue)),
        )
        return result


def color_histogram(image: ByteImage, bins: int = 32) -> FloatArray:
    histograms = []
    for channel in range(3):
        histogram, _ = np.histogram(image[..., channel], bins=bins, range=(0, 256), density=True)
        histograms.append(histogram)
    return np.concatenate(histograms).astype(np.float64)


def histogram_distance(first: ByteImage, second: ByteImage, bins: int = 32) -> float:
    first_histogram = color_histogram(first, bins)
    second_histogram = color_histogram(second, bins)
    numerator = np.square(first_histogram - second_histogram)
    denominator = first_histogram + second_histogram + 1e-12
    return float(0.5 * np.sum(numerator / denominator))


def structural_similarity(first: ByteImage, second: ByteImage) -> float:
    if first.shape != second.shape:
        raise ValueError("images must have equal shapes")
    first_values = first.astype(np.float64)
    second_values = second.astype(np.float64)
    first_mean = first_values.mean()
    second_mean = second_values.mean()
    first_variance = first_values.var()
    second_variance = second_values.var()
    covariance = np.mean((first_values - first_mean) * (second_values - second_mean))
    first_constant = (0.01 * 255.0) ** 2
    second_constant = (0.03 * 255.0) ** 2
    luminance = (2.0 * first_mean * second_mean + first_constant) / (
        first_mean**2 + second_mean**2 + first_constant
    )
    structure = (2.0 * covariance + second_constant) / (
        first_variance + second_variance + second_constant
    )
    return float(luminance * structure)


def rotation_matrix(angle_degrees: float, width: int, height: int) -> FloatArray:
    center = (width / 2.0, height / 2.0)
    return cv2.getRotationMatrix2D(center, angle_degrees, 1.0).astype(np.float64)


def arbitrary_rotation(image: ByteImage, angle_degrees: float) -> ByteImage:
    height, width = image.shape[:2]
    matrix = rotation_matrix(angle_degrees, width, height)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    output_width = int(height * sine + width * cosine)
    output_height = int(height * cosine + width * sine)
    matrix[0, 2] += output_width / 2.0 - width / 2.0
    matrix[1, 2] += output_height / 2.0 - height / 2.0
    rotated = cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    start_x = max(0, (output_width - width) // 2)
    start_y = max(0, (output_height - height) // 2)
    cropped = rotated[start_y : start_y + height, start_x : start_x + width]
    if cropped.shape[:2] != (height, width):
        cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    return cropped


def gaussian_blur(image: ByteImage, sigma: float) -> ByteImage:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel = 2 * radius + 1
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
