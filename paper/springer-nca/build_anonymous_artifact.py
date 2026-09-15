#!/usr/bin/env python3
"""Build an anonymous, lightweight code-and-figure artifact for review."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUTPUT = ROOT / "build"
STAGING = OUTPUT / "anonymous-code-artifact"
ARCHIVE = OUTPUT / "nca-anonymous-code-artifact.zip"

ANONYMITY_MARKERS = (
    "王天熙",
    "Wang Tianxi",
    "Tianxi Wang",
    "augustuswtx",
    "/Users/",
    "/home/",
    "shu-common-",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_sources() -> list[Path]:
    sources: list[Path] = []
    for path in (REPO / "MFON").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            if path.suffix in {".py", ".yaml", ".yml"} or path.name == "README.md":
                sources.append(path)
    for base in (REPO / "paper" / "figures" / "data", REPO / "paper" / "figures" / "scripts"):
        for path in base.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                if path.suffix in {".csv", ".json", ".py"}:
                    sources.append(path)
    return sorted(set(sources))


def check_text_anonymity(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = [marker for marker in ANONYMITY_MARKERS if marker.casefold() in text.casefold()]
    if hits:
        raise RuntimeError(f"anonymity audit failed for {path}: {hits}")


def build() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    copied: list[Path] = []
    for source in selected_sources():
        if not source.is_file():
            raise FileNotFoundError(source)
        check_text_anonymity(source)
        destination = STAGING / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    readme = STAGING / "ARTIFACT_README.md"
    readme.write_text(
        "# Anonymous review artifact\n\n"
        "This archive contains the MFON/P4 implementation, unit tests, read-only "
        "audit programs, frozen figure source values, and Python figure scripts.\n\n"
        "It intentionally excludes CMU-MOSI/CMU-MOSEI data, BERT files, checkpoints, "
        "logs, credentials, git history, and personal attachments. Obtain the benchmark "
        "features through their official access process, then configure local paths in "
        "the dataset configuration files.\n\n"
        "Run contract tests from `MFON/` with:\n\n"
        "```bash\npython -m unittest discover -s tests -v\n```\n\n"
        "Rebuild manuscript figures from the repository root with:\n\n"
        "```bash\npython3 paper/figures/scripts/make_f1_method_overview.py\n"
        "python3 paper/figures/scripts/make_quantitative_figures.py\n```\n\n"
        "The cross-sample audit is read-only and requires an existing frozen checkpoint. "
        "Use `python audit_cross_sample_validity.py --help` from `MFON/` for the "
        "available arguments. Its auxiliary-loss association is a target-fidelity proxy, "
        "not proof of a perceptual quality scale. The completed per-seed MOSEI "
        "validation results are recorded in "
        "`paper/figures/data/c1_mosei_cross_sample_validity.csv`; they reject visual "
        "cross-sample KL fidelity and show only weak acoustic association.\n",
        encoding="utf-8",
    )
    copied.append(readme)

    manifest = {
        "artifact": "anonymous code, audit, and figure-source package",
        "exclusions": [
            "datasets",
            "pretrained language-model files",
            "checkpoints",
            "logs",
            "credentials",
            "git history",
            "personal attachments",
        ],
        "files": {
            str(path.relative_to(STAGING)): sha256(path)
            for path in sorted(copied)
        },
    }
    manifest_path = STAGING / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    check_text_anonymity(manifest_path)

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(STAGING))

    print(f"anonymous artifact: {STAGING}")
    print(f"artifact archive: {ARCHIVE}")
    print(f"files: {len(list(STAGING.rglob('*.*')))}")


if __name__ == "__main__":
    build()
