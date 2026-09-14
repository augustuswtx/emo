#!/usr/bin/env python3
"""Build a flat, anonymous Springer Nature submission package for NCA."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
FIGURES = {
    "../figures/f1_method_overview_en.pdf": "Fig1.pdf",
    "../figures/f2_mosei_main_results_en.pdf": "Fig2.pdf",
    "../figures/f3_mosei_gaussian_audit_en.pdf": "Fig3.pdf",
    "../figures/f4_mosei_cross_corruption_en.pdf": "Fig4.pdf",
    "../figures/f5_mosei_efficiency_en.pdf": "Fig5.pdf",
}
ANONYMITY_MARKERS = (
    "王天熙",
    "Wang Tianxi",
    "Tianxi Wang",
    "augustuswtx",
    "/Users/",
    "/home/jovyan/",
    "shu-common-",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_manuscript() -> str:
    wrapper = (ROOT / "main.tex").read_text(encoding="utf-8")
    body = (ROOT / "body.tex").read_text(encoding="utf-8")
    marker = r"\input{body}"
    if wrapper.count(marker) != 1:
        raise RuntimeError("main.tex must contain exactly one \\input{body} marker")
    merged = wrapper.replace(marker, body)
    merged = merged.replace(
        r"\bibliography{../../docs/small-paper-references}",
        r"\bibliography{references}",
    )
    for old, new in FIGURES.items():
        if old not in merged:
            raise RuntimeError(f"expected figure reference missing: {old}")
        merged = merged.replace(old, new)
    if r"\input{" in merged or r"\include{" in merged:
        raise RuntimeError("generated manuscript must be a single TeX document")
    return merged


def check_anonymity(paths: list[Path]) -> None:
    findings: list[str] = []
    for path in paths:
        if path.suffix.lower() not in {".tex", ".bib", ".txt", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in ANONYMITY_MARKERS:
            if marker.casefold() in text.casefold():
                findings.append(f"{path.name}: {marker}")
    if findings:
        raise RuntimeError("anonymity audit failed: " + "; ".join(findings))


def make_zip(source_dir: Path, destination: Path) -> None:
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)


def build(output_root: Path) -> tuple[Path, Path]:
    manuscript_dir = output_root / "anonymous-manuscript"
    if manuscript_dir.exists():
        shutil.rmtree(manuscript_dir)
    manuscript_dir.mkdir(parents=True)

    (manuscript_dir / "main.tex").write_text(
        merge_manuscript(), encoding="utf-8"
    )

    copies = {
        ROOT / "sn-jnl.cls": manuscript_dir / "sn-jnl.cls",
        ROOT / "sn-basic.bst": manuscript_dir / "sn-basic.bst",
        REPO / "docs" / "small-paper-references.bib": manuscript_dir / "references.bib",
    }
    for old, new in FIGURES.items():
        copies[(ROOT / old).resolve()] = manuscript_dir / new

    for source, destination in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)

    source_files = sorted(path for path in manuscript_dir.iterdir() if path.is_file())
    check_anonymity(source_files)
    manifest = {
        "journal": "Neural Computing and Applications",
        "review_package": "anonymous manuscript",
        "build_command": "pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex",
        "files": {path.name: sha256(path) for path in source_files},
    }
    manifest_path = output_root / "anonymous-manuscript-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    zip_path = output_root / "nca-anonymous-manuscript.zip"
    if zip_path.exists():
        zip_path.unlink()
    make_zip(manuscript_dir, zip_path)
    return manuscript_dir, zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build",
        help="output directory (default: paper/springer-nca/build)",
    )
    args = parser.parse_args()
    manuscript_dir, zip_path = build(args.output.resolve())
    print(f"anonymous manuscript: {manuscript_dir}")
    print(f"submission archive: {zip_path}")


if __name__ == "__main__":
    main()
