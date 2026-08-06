import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

from dgvlm_equitable_glioma.cohorts.harmonization import (
    RawMolecularProfile,
    harmonize,
    valid_profile,
)

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="dgvlm-prepare")
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--slide-root", type=Path, required=True)
    result.add_argument("--domain", type=str, required=True)
    result.add_argument("--slide-id-column", default="slide_id")
    result.add_argument("--patient-id-column", default="patient_id")
    result.add_argument("--slide-path-column", default="slide_path")
    result.add_argument("--idh1-column", default="idh1")
    result.add_argument("--idh2-column", default="idh2")
    result.add_argument("--codeletion-column", default="codeletion_1p19q")
    result.add_argument("--mgmt-column", default="mgmt")
    result.add_argument("--histology-column", default="histology")
    result.add_argument("--grade-column", default="grade")
    result.add_argument("--require-slides", action="store_true")
    return result


def optional(row: dict[str, str], column: str) -> str | None:
    value = row.get(column)
    if value is None or not value.strip():
        return None
    return value


def output_row(
    row: dict[str, str],
    arguments: argparse.Namespace,
) -> tuple[dict[str, object], tuple[str, ...]] | None:
    slide_id = row[arguments.slide_id_column].strip()
    relative_path = Path(row[arguments.slide_path_column].strip())
    slide_path = (
        relative_path if relative_path.is_absolute() else arguments.slide_root / relative_path
    )
    if arguments.require_slides and not slide_path.is_file():
        return None
    raw = RawMolecularProfile(
        idh1=optional(row, arguments.idh1_column),
        idh2=optional(row, arguments.idh2_column),
        codeletion_1p19q=optional(row, arguments.codeletion_column),
        mgmt=optional(row, arguments.mgmt_column),
        histology=optional(row, arguments.histology_column),
        grade=optional(row, arguments.grade_column),
    )
    profile = harmonize(raw)
    if not valid_profile(profile):
        return None
    prepared: dict[str, object] = {
        "slide_id": slide_id,
        "patient_id": row.get(arguments.patient_id_column, slide_id).strip() or slide_id,
        "domain": arguments.domain,
        "slide_path": str(slide_path),
        "idh": "" if profile.idh is None else profile.idh,
        "codeletion": "" if profile.codeletion is None else profile.codeletion,
        "mgmt": "" if profile.mgmt is None else profile.mgmt,
        "subtype": "" if profile.subtype is None else profile.subtype,
    }
    return prepared, profile.reasons


def run(arguments: argparse.Namespace) -> None:
    with arguments.input.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        rows = [dict(row) for row in reader]
    prepared_rows: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    rejected = 0
    for row in rows:
        prepared = output_row(row, arguments)
        if prepared is None:
            rejected += 1
            continue
        output, reasons = prepared
        prepared_rows.append(output)
        reason_counts.update(reasons)
    identifiers = [str(row["slide_id"]) for row in prepared_rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("prepared slide identifiers are not unique")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
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
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prepared_rows)
    temporary.replace(arguments.output)
    LOGGER.info(
        "prepared=%d rejected=%d missing=%s",
        len(prepared_rows),
        rejected,
        dict(reason_counts),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(parser().parse_args())


if __name__ == "__main__":
    main()
