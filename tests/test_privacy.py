"""Privacy scan tests: the public tree must stay clean."""

from __future__ import annotations

from pathlib import Path

from app.privacy import FORBIDDEN_TERMS, Finding, scan_tree, summarise

ROOT = Path(__file__).resolve().parents[1]


def test_work_tree_is_clean() -> None:
    findings = scan_tree(ROOT)
    assert findings == [], summarise(findings)


def test_forbidden_terms_are_not_plain_text() -> None:
    """The scanner must not ship the markers it looks for."""
    text = (ROOT / "app" / "privacy.py").read_text(encoding="utf-8")
    for term in FORBIDDEN_TERMS:
        assert term not in text
    assert len(FORBIDDEN_TERMS) >= 10


def test_scan_catches_a_leak(tmp_path: Path) -> None:
    leak = tmp_path / "notes.md"
    leak.write_text("Source: " + FORBIDDEN_TERMS[0] + " study\n", encoding="utf-8")
    findings = scan_tree(tmp_path)
    assert any(finding.kind == "forbidden term" for finding in findings)


def test_scan_catches_dashes(tmp_path: Path) -> None:
    page = tmp_path / "page.html"
    page.write_text("a " + chr(0x2014) + " b " + chr(0x2013) + " c", encoding="utf-8")
    findings = scan_tree(tmp_path)
    assert sum(1 for finding in findings if finding.kind == "forbidden dash") == 2


def test_scan_catches_patient_id_column(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("Patient ID,Vessel\n1,1\n", encoding="utf-8")
    findings = scan_tree(tmp_path)
    kinds = {finding.kind for finding in findings}
    assert "forbidden column" in kinds


def test_scan_accepts_record_id_column(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("Record ID,Vessel\n1,1\n", encoding="utf-8")
    assert scan_tree(tmp_path) == []


def test_scan_catches_credentials(tmp_path: Path) -> None:
    env = tmp_path / "notes.txt"
    env.write_text("token " + "sk-" + "a" * 24 + "\n", encoding="utf-8")
    findings = scan_tree(tmp_path)
    assert any(finding.kind == "api key" for finding in findings)


def test_scan_allows_placeholder_email(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("Contact you@example.com for questions.\n", encoding="utf-8")
    assert scan_tree(tmp_path) == []


def test_finding_renders_location() -> None:
    finding = Finding("docs/a.md", 7, "forbidden term", "x")
    assert finding.render() == "docs/a.md:7: forbidden term: x"
