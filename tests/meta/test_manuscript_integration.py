"""Phase 5: manuscript macros, drift detection, bibliography and citation provenance."""

from __future__ import annotations

import copy
import json
import re

import pytest

from qsarena.meta_analysis import io, text

META_NUMBERS = io.MANUSCRIPT_ASSETS / "meta_numbers.json"
BIB = io.REPO_ROOT / "submission" / "references.bib"
NEW_BIB_KEYS = ["olier2018metaqsar", "sheridan2004similarity", "sheridan2015relative", "chen2025datascaling"]
PLACEHOLDER_TOKENS = ["COMPLETE FROM DOI", "TODO verify", "AUTHOR_PLACEHOLDER", "Anonymous and Others"]


@pytest.fixture(scope="module")
def numbers():
    return json.loads(META_NUMBERS.read_text(encoding="utf-8"))


def _bib_entries() -> dict[str, str]:
    raw = BIB.read_text(encoding="utf-8")
    entries = {}
    for match in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", raw, re.DOTALL):
        entries[match.group(1).strip()] = match.group(2)
    return entries


def test_every_macro_resolves(numbers):
    for name, template in text.BLOCKS.items():
        assert text.unresolved(template, numbers["macros"]) == [], name
        cites = {m.split(":", 1)[1] for m in text.MACRO.findall(template) if m.startswith("cite:")}
        assert cites <= set(text.CITATIONS), cites - set(text.CITATIONS)


def test_manuscript_blocks_match_meta_numbers(numbers):
    assert text.check_manuscript(numbers) == []


def test_perturbed_number_is_detected(numbers):
    perturbed = copy.deepcopy(numbers)
    perturbed["macros"]["lodo_bacc"] = "0.99"
    assert any("drifted" in problem for problem in text.check_manuscript(perturbed))


def test_verifier_runs_meta_checks():
    source = (io.REPO_ROOT / "portable_colab_qsar_bundle" / "verify_manuscript_numbers.py").read_text(encoding="utf-8")
    assert "meta_numbers.json" in source and "check_manuscript" in source


def test_new_bib_keys_exist_and_are_cited():
    entries = _bib_entries()
    body = (io.REPO_ROOT / "submission" / "body.tex").read_text(encoding="utf-8")
    cited = {k.strip() for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", body) for k in group.split(",")}
    for key in NEW_BIB_KEYS:
        assert key in entries, key
        assert key in cited, f"{key} is never cited in body.tex"
    assert cited <= set(entries), f"cited but missing from references.bib: {sorted(cited - set(entries))}"


def test_markdown_reference_numbers_match_citation_map():
    md = (io.REPO_ROOT / "manuscript.md").read_text(encoding="utf-8")
    references = md.split("## References", 1)[1]
    expected_titles = {
        67: "Meta-QSAR",
        68: "Similarity to molecules in the training set",
        69: "relative importance of domain applicability metrics",
        70: "Data scaling and generalization insights",
        16: "Benchmarking ML in ADMET predictions",
        46: "systematic study of key elements",
        37: "Analyzing learned molecular representations",
    }
    for number, title in expected_titles.items():
        line = re.search(rf"^{number}\. (.*)$", references, re.MULTILINE)
        assert line and title.lower() in line.group(1).lower(), (number, title)
    for key, number in text.CITATIONS.items():
        assert re.search(rf"^{number}\. ", references, re.MULTILINE), key


def test_no_unresolved_author_placeholders():
    entries = _bib_entries()
    for key in NEW_BIB_KEYS:
        entry = entries[key]
        for token in PLACEHOLDER_TOKENS:
            assert token.lower() not in entry.lower(), f"{key} still contains {token!r}"
        assert re.search(r"author\s*=\s*\{[^}]*[A-Za-z]", entry), key
        assert re.search(r"doi\s*=\s*\{10\.", entry), key
