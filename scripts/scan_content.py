#!/usr/bin/env python3
"""Scan the public work tree for material that must never ship.

Checks for institution names, original data file names, personal data
patterns, credentials, Malay leftovers and forbidden dashes. Exits 0 only when
the tree is clean.

Usage::

    python scripts/scan_content.py                    # scan the repo
    python scripts/scan_content.py --root some/path    # scan another tree
    python scripts/scan_content.py --out /tmp/scan.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.privacy import scan_tree  # noqa: E402

MALAY_TERMS = (
    "dan ",
    "yang ",
    "adalah",
    "dengan ",
    "untuk ",
    "tidak ",
    "kesimpulan",
    "rumusan",
    "pesakit",
    "jantung",
    "rawatan",
)


def malay_findings(root: Path) -> list[str]:
    """Crude Malay leftover check on prose files only."""
    from app.privacy import iter_text_files, read_text

    hits: list[str] = []
    for path in iter_text_files(root):
        if path.suffix.lower() not in {".md", ".txt", ".html"}:
            continue
        for number, line in enumerate(read_text(path).splitlines(), start=1):
            lowered = f" {line.lower()} "
            for term in MALAY_TERMS:
                if f" {term} " in lowered and not line.strip().startswith(("#", "-", "|")):
                    hits.append(f"{path.relative_to(root)}:{number}: malay leftover: {term.strip()}")
                    break
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=Path("/tmp/ctaid-poc-scan.txt"))
    args = parser.parse_args(argv)

    root = args.root.resolve()
    findings = scan_tree(root)
    malay = malay_findings(root)

    lines = ["# CTAID POC content scan", f"root: {root}", "files scanned: work tree only", ""]
    if findings:
        lines.append(f"forbidden matches: {len(findings)}")
        lines.extend(finding.render() for finding in findings)
    else:
        lines.append("forbidden matches: 0")
    if malay:
        lines.append("")
        lines.append(f"malay leftovers: {len(malay)}")
        lines.extend(malay)
    else:
        lines.append("malay leftovers: 0")

    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0 if not findings and not malay else 1


if __name__ == "__main__":
    raise SystemExit(main())
