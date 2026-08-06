import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class RuntimeAudit:
    python: str
    platform: str
    torch: str
    cuda_runtime: str | None
    cudnn: int | None
    gpu_names: tuple[str, ...]
    packages: Mapping[str, str]
    source_revision: str | None


def file_digest(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def manifest_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(file_digest(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def source_revision(directory: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return None
    return process.stdout.strip()


def runtime_audit(directory: Path) -> RuntimeAudit:
    gpu_names = tuple(
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    )
    packages = package_versions(
        (
            "torch",
            "torchvision",
            "numpy",
            "scipy",
            "scikit-learn",
            "pandas",
            "omegaconf",
            "hydra-core",
            "openslide-python",
            "opencv-python-headless",
            "h5py",
            "lifelines",
        )
    )
    return RuntimeAudit(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu_names=gpu_names,
        packages=packages,
        source_revision=source_revision(directory),
    )


def write_audit(path: Path, audit: RuntimeAudit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(asdict(audit), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
