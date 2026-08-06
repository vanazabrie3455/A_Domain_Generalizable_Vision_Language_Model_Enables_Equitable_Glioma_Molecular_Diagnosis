import re
from dataclasses import dataclass
from typing import Literal

MolecularState = Literal[0, 1]
IntegratedSubtype = Literal[0, 1, 2]


@dataclass(frozen=True)
class RawMolecularProfile:
    idh1: str | None
    idh2: str | None
    codeletion_1p19q: str | None
    mgmt: str | None
    histology: str | None
    grade: str | None


@dataclass(frozen=True)
class HarmonizedProfile:
    idh: MolecularState | None
    codeletion: MolecularState | None
    mgmt: MolecularState | None
    subtype: IntegratedSubtype | None
    grade: int | None
    reasons: tuple[str, ...]


POSITIVE_TERMS = frozenset(
    {
        "positive",
        "pos",
        "mutant",
        "mutated",
        "mutation",
        "detected",
        "yes",
        "true",
        "1",
        "methylated",
        "codeleted",
        "co-deleted",
        "loss",
    }
)

NEGATIVE_TERMS = frozenset(
    {
        "negative",
        "neg",
        "wildtype",
        "wild-type",
        "wt",
        "not detected",
        "no",
        "false",
        "0",
        "unmethylated",
        "intact",
        "retained",
    }
)

MISSING_TERMS = frozenset(
    {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "unknown",
        "not available",
        "not assessed",
        "indeterminate",
        "equivocal",
    }
)


def normalized(value: str | None) -> str:
    if value is None:
        return ""
    compact = re.sub(r"\s+", " ", value.strip().lower())
    return compact.replace("_", " ")


def molecular_state(value: str | None) -> MolecularState | None:
    term = normalized(value)
    if term in MISSING_TERMS:
        return None
    if term in POSITIVE_TERMS:
        return 1
    if term in NEGATIVE_TERMS:
        return 0
    positive_fragments = ("mut", "methyl", "codelet", "co-delet")
    negative_fragments = ("wild", "unmethyl", "intact", "retain")
    if any(fragment in term for fragment in negative_fragments):
        return 0
    if any(fragment in term for fragment in positive_fragments):
        return 1
    return None


def idh_state(idh1: str | None, idh2: str | None) -> MolecularState | None:
    first = molecular_state(idh1)
    second = molecular_state(idh2)
    if first == 1 or second == 1:
        return 1
    if first == 0 and second in {0, None}:
        return 0
    if second == 0 and first in {0, None}:
        return 0
    return None


def grade_value(value: str | None) -> int | None:
    term = normalized(value)
    if term in MISSING_TERMS:
        return None
    matches = re.findall(r"\b[234]\b", term)
    if matches:
        return int(matches[-1])
    roman = {"ii": 2, "iii": 3, "iv": 4}
    for token, grade in roman.items():
        if re.search(rf"\b{token}\b", term):
            return grade
    return None


def integrated_subtype(
    idh: MolecularState | None,
    codeletion: MolecularState | None,
) -> IntegratedSubtype | None:
    if idh == 0:
        return 2
    if idh == 1 and codeletion == 1:
        return 1
    if idh == 1 and codeletion == 0:
        return 0
    return None


def harmonize(profile: RawMolecularProfile) -> HarmonizedProfile:
    reasons: list[str] = []
    idh = idh_state(profile.idh1, profile.idh2)
    codeletion = molecular_state(profile.codeletion_1p19q)
    mgmt = molecular_state(profile.mgmt)
    grade = grade_value(profile.grade)
    subtype = integrated_subtype(idh, codeletion)
    if idh is None:
        reasons.append("IDH status unavailable or indeterminate")
    if codeletion is None:
        reasons.append("1p/19q status unavailable or indeterminate")
    if mgmt is None:
        reasons.append("MGMT status unavailable or indeterminate")
    if subtype is None:
        reasons.append("integrated subtype requires IDH and 1p/19q")
    if grade is None:
        reasons.append("WHO grade unavailable or indeterminate")
    return HarmonizedProfile(idh, codeletion, mgmt, subtype, grade, tuple(reasons))


def valid_profile(profile: HarmonizedProfile) -> bool:
    if profile.idh == 0 and profile.codeletion == 1:
        return False
    if profile.subtype == 0 and not (profile.idh == 1 and profile.codeletion == 0):
        return False
    if profile.subtype == 1 and not (profile.idh == 1 and profile.codeletion == 1):
        return False
    return not (profile.subtype == 2 and profile.idh != 0)


def label_dictionary(profile: HarmonizedProfile) -> dict[str, int]:
    values = {
        "idh": profile.idh,
        "codeletion": profile.codeletion,
        "mgmt": profile.mgmt,
        "subtype": profile.subtype,
    }
    return {name: value for name, value in values.items() if value is not None}
