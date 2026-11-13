"""Privacy and content scan helpers.

Used by the test suite and by ``scripts/scan_content.py``. The scan walks the
public work tree only: no private reference folder, no virtualenv, no git
internals.

The forbidden term list is stored base64 encoded. That is deliberate: a public
repository should not carry the very strings the scanner is meant to catch, so
a plain text search over the repo stays useful for reviewers.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "ref",
    "poc-rebuild-20261113",
    "var",
    "htmlcov",
    ".streamlit",
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".css",
    ".html",
    ".js",
    ".ts",
    ".svg",
    ".ini",
    ".cfg",
    ".conf",
    ".webmanifest",
    ".csv",
}

SPECIAL_NAMES = {"Dockerfile", "LICENSE", "Makefile", "Procfile", ".gitignore", ".dockerignore"}

# Base64 encoded markers: institution names, private project names and original
# source file names that must never appear in the public tree.
_TERMS_B64 = (
    "SGFubmFu",  # institution
    "TWVkaXNwYQ==",  # clinic name
    "VVBOTQ==",  # university abbreviation
    "VW5pdmVyc2l0aQ==",  # university word
    "VU5JU0VM",  # university abbreviation
    "U2VsYW5nb3I=",  # state name
    "RkZNUQ==",  # questionnaire acronym
    "TUxfQWxs",  # original dataset name
    "c2FtcGxlX2RhdGEv",  # original sample folder
    "VFBPVHBpcGU=",  # original model file name
    "cGhvdG9faGVhcnQ=",  # original image file name
    "cGhvdG9fZmxvdy1lbg==",  # original image file name
    "TWFyZGhpeWF0aQ==",  # original presenter name
    "U3BhIE1pbmRh",  # private project name
    "VE5C",  # private organisation
    "Q1ZhenJpbA==",  # private folder name
    "TmFkaWE=",  # private folder name
)

FORBIDDEN_TERMS = tuple(base64.b64decode(item).decode("utf-8") for item in _TERMS_B64)

# Patterns that would indicate leaked personal data or credentials.
FORBIDDEN_PATTERNS = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"), "email address"),
    (re.compile(r"(?<!\d)(?:\+?60|0)1\d[- ]?\d{3,4}[- ]?\d{3,4}(?!\d)"), "phone number"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "api key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws access key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "github token"),
)

# Placeholder addresses that docs are allowed to mention.
EMAIL_ALLOWLIST = {"noreply@azril.io", "you@example.com"}

DASH_CHARS = {"\u2013": "en dash", "\u2014": "em dash"}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    detail: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.detail}"


def iter_text_files(root: Path):
    root = Path(root).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in SPECIAL_NAMES or path.suffix.lower() == "" and path.name.startswith("Dockerfile"):
            yield path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def scan_file(path: Path, root: Path) -> list[Finding]:
    text = read_text(path)
    relative = str(path.relative_to(root))
    findings: list[Finding] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for term in FORBIDDEN_TERMS:
            if term.lower() in lowered:
                findings.append(Finding(relative, line_number, "forbidden term", term))
        for character, name in DASH_CHARS.items():
            if character in line:
                findings.append(Finding(relative, line_number, "forbidden dash", name))
        for pattern, label in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(line):
                value = match.group(0)
                if label == "email address" and (value in EMAIL_ALLOWLIST or value.endswith("@example.com")):
                    continue
                findings.append(Finding(relative, line_number, label, value[:40]))

    # Column rule: committed data must carry Record ID, never Patient ID.
    if path.suffix.lower() == ".csv":
        header = text.splitlines()[0] if text.splitlines() else ""
        if "Patient ID" in header:
            findings.append(Finding(relative, 1, "forbidden column", "Patient ID"))
        if "Record ID" not in header:
            findings.append(Finding(relative, 1, "missing column", "Record ID"))

    return findings


def scan_tree(root: Path) -> list[Finding]:
    root = Path(root).resolve()
    findings: list[Finding] = []
    for path in iter_text_files(root):
        findings.extend(scan_file(path, root))
    return findings


def summarise(findings: list[Finding]) -> str:
    if not findings:
        return "clean: no forbidden terms, dashes, personal data or credentials found"
    return "\n".join(finding.render() for finding in findings)
