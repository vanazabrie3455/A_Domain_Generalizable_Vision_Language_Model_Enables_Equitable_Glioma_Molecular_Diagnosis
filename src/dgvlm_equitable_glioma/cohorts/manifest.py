import csv
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from dgvlm_equitable_glioma.types import SlideRecord, TargetName


@dataclass(frozen=True)
class Manifest:
    records: tuple[SlideRecord, ...]
    digest: str

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[SlideRecord]:
        return iter(self.records)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({record.domain for record in self.records}))

    def select_domains(self, domains: Iterable[str]) -> "Manifest":
        selected = frozenset(domains)
        records = tuple(record for record in self.records if record.domain in selected)
        return Manifest(records=records, digest=_records_digest(records))

    def exclude_domains(self, domains: Iterable[str]) -> "Manifest":
        excluded = frozenset(domains)
        records = tuple(record for record in self.records if record.domain not in excluded)
        return Manifest(records=records, digest=_records_digest(records))

    def target_records(self, target: TargetName) -> tuple[SlideRecord, ...]:
        return tuple(record for record in self.records if target in record.available)

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for record in self.records:
            result[record.domain] = result.get(record.domain, 0) + 1
        return result


def _canonical_record(record: SlideRecord) -> bytes:
    content = {
        "slide_id": record.slide_id,
        "patient_id": record.patient_id,
        "domain": record.domain,
        "path": str(record.path),
        "labels": dict(sorted(record.labels.items())),
        "available": sorted(record.available),
    }
    return json.dumps(content, separators=(",", ":"), sort_keys=True).encode()


def _records_digest(records: Iterable[SlideRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.slide_id):
        digest.update(_canonical_record(record))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "" or stripped.lower() in {"na", "nan", "none", "unknown"}:
        return None
    return int(stripped)


def _record_from_row(row: Mapping[str, str], slide_root: Path) -> SlideRecord:
    targets: tuple[TargetName, ...] = ("idh", "codeletion", "mgmt", "subtype")
    labels: dict[TargetName, int] = {}
    available: set[TargetName] = set()
    for target in targets:
        value = _parse_optional_int(row.get(target))
        if value is not None:
            labels[target] = value
            available.add(target)
    relative = Path(row["slide_path"])
    path = relative if relative.is_absolute() else slide_root / relative
    return SlideRecord(
        slide_id=row["slide_id"],
        patient_id=row.get("patient_id", row["slide_id"]),
        domain=row["domain"],
        path=path,
        labels=labels,
        available=frozenset(available),
    )


def load_manifest(path: Path, slide_root: Path | None = None) -> Manifest:
    root = path.parent if slide_root is None else slide_root
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"slide_id", "domain", "slide_path"}
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"manifest is missing columns: {names}")
        records = tuple(_record_from_row(row, root) for row in reader)
    ids = [record.slide_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest slide identifiers must be unique")
    return Manifest(records=records, digest=_records_digest(records))


def write_manifest(manifest: Manifest, path: Path) -> None:
    fieldnames = [
        "slide_id",
        "patient_id",
        "domain",
        "slide_path",
        "idh",
        "codeletion",
        "mgmt",
        "subtype",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in manifest.records:
            row: dict[str, object] = {
                "slide_id": record.slide_id,
                "patient_id": record.patient_id,
                "domain": record.domain,
                "slide_path": str(record.path),
            }
            row.update({target: record.labels.get(target, "") for target in fieldnames[4:]})
            writer.writerow(row)
    temporary.replace(path)
