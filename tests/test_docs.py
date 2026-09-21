from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mkdocs_navigation_targets_exist() -> None:
    config = yaml.safe_load((REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    targets: list[str] = []

    def collect(items: list[object]) -> None:
        for item in items:
            if isinstance(item, str):
                targets.append(item)
            elif isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, str):
                        targets.append(value)
                    elif isinstance(value, list):
                        collect(value)

    collect(config["nav"])
    missing = [target for target in targets if not (REPO_ROOT / "docs" / target).is_file()]
    assert missing == []


def test_public_documentation_local_links_exist() -> None:
    documents = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
    ]
    missing: list[str] = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for raw_target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text):
            target = raw_target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(REPO_ROOT)} -> {raw_target}")

    assert missing == []


def test_public_documentation_excludes_internal_project_history() -> None:
    documents = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs").rglob("*.md")),
    ]
    forbidden = re.compile(
        r"AO Predict|ao[-_]predict|girmos(?:-aosims)?|"
        r"extraction plan|migration plan|phase \d+",
        re.IGNORECASE,
    )
    matches = [
        f"{document.relative_to(REPO_ROOT)}: {match.group(0)}"
        for document in documents
        for match in forbidden.finditer(document.read_text(encoding="utf-8"))
    ]

    assert matches == []


def test_generated_reference_uses_canonical_package_boundary() -> None:
    reference = (REPO_ROOT / "docs/reference/api.md").read_text(encoding="utf-8")

    assert reference.strip() == "# Python API\n\n::: hybrid_ao_psf"
