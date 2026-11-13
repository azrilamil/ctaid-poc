from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANNED = (
    "UNISEL",
    "Universiti Selangor",
    "ML_All",
    "Hannan",
    "Medispa",
    "Selamat",
)
SKIP = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", "models"}
TEXT_SUFFIX = {
    ".py", ".md", ".txt", ".html", ".css", ".js", ".yml", ".yaml",
    ".json", ".svg", ".toml", ".cfg", ".conf", ".webmanifest",
}


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP for part in p.parts):
            continue
        if p.suffix.lower() in TEXT_SUFFIX or p.name in {"Dockerfile", "LICENSE"}:
            yield p


def test_no_banned_terms_or_dashes_in_product_copy():
    hits = []
    for p in iter_files():
        if p.suffix == ".py" and p.name.startswith("test_"):
            continue
        text = p.read_text(encoding="utf-8")
        # Skip this test module's own patterns by excluding tests/.
        if "tests/" in str(p).replace("\\", "/"):
            continue
        rel = str(p.relative_to(ROOT))
        for term in BANNED:
            if term.lower() in text.lower():
                hits.append(f"{rel}: {term}")
        if chr(0x2014) in text or chr(0x2013) in text:
            hits.append(f"{rel}: dash")
        if "ML_All features" in text:
            hits.append(f"{rel}: ML_All features")
    assert hits == []
