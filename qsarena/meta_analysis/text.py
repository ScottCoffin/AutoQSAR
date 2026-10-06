"""Manuscript text for the meta-analysis, rendered from ``meta_numbers.json`` macros.

Every quantitative claim is a ``{{macro}}`` placeholder filled from ``meta_numbers.json["macros"]``;
conditional wording (e.g. whether a trend narrows) is decided in ``pipeline.build_numbers`` and
arrives as a macro too. The rendered text lives between marker comments in both manuscript formats:

- ``manuscript.md``:  ``<!-- META:<block> -->`` ... ``<!-- /META -->``
- ``submission/body.tex``: ``% META:<block>`` ... ``% /META`` (main-text blocks)
- ``submission/additional_file_1.tex``: same markers (supplementary blocks, ``BLOCK_TARGETS[name] == "si"``)

Never hand-edit inside those markers. ``verify_manuscript_numbers.py`` re-renders the blocks from
``meta_numbers.json`` and fails if the manuscript drifts from the numbers.

Since revision R1 (2026-10) the full meta-analysis is Supplementary Note S4 of Additional file 1 (block
``section_3_14``, kept under its historical name) and the main text carries a short summary
(``section_meta_summary``, Section 3.14). The nested-selection evidence moved from the Methods to its own
Results subsection (``results_nested_selection``, Section 3.4); the Methods block keeps the procedure.
"""

from __future__ import annotations

import re
from pathlib import Path

from qsarena.meta_analysis.io import REPO_ROOT

MANUSCRIPT_MD = REPO_ROOT / "manuscript.md"
BODY_TEX = REPO_ROOT / "submission" / "body.tex"
SI_TEX = REPO_ROOT / "submission" / "additional_file_1.tex"
MACRO = re.compile(r"\{\{([A-Za-z0-9_:]+)\}\}")

#: Citation key -> number in manuscript.md's numbered reference list (body.tex uses \cite{key}).
CITATIONS = {
    "wu2018moleculenet": 4,
    "xia2023understanding": 10,
    "kamuntavicius2025benchmarking": 16,
    "yang2019analyzing": 37,
    "deng2023systematic": 46,
    "olier2018metaqsar": 67,
    "sheridan2004similarity": 68,
    "sheridan2015relative": 69,
    "chen2025datascaling": 70,
}

#: Full meta-analysis: Supplementary Note S4 in Additional file 1 (and the Markdown supplement).
SECTION = """\
### Supplementary Note S4. When does each family win? A dataset-property meta-analysis

Section 3.2 of the main text found that no model family dominates. Here we ask when each family comes close to \
the best, using properties of the dataset alone. This is per-dataset algorithm selection from dataset \
meta-features, the approach Meta-QSAR applied to thousands of QSAR problems {{cite:olier2018metaqsar}}, and it \
addresses the observation that the best ADMET model and representation are strongly dataset-dependent \
{{cite:kamuntavicius2025benchmarking}}. For each of the {{n_datasets}} datasets we computed meta-features on the \
exact train/test partition used in the benchmark, verified against the recorded split hashes (Table S9). They are: \
training-set size, label imbalance or skew, Bemis–Murcko scaffold diversity, internal fingerprint diversity, and \
train-to-test similarity. Similarity is summarised by SNN, the Tanimoto similarity of each test molecule to its \
nearest training molecule, which is known to track prediction error {{cite:sheridan2004similarity}}. Winners are \
single-split, single-seed outcomes, so the outcome modelled is the continuous relative gap of Figure S5 (Note S6), \
never the identity of the winner. Every interval is a 95% percentile bootstrap over datasets.

**Training-set size.** {{size_sentence}} For the pre-registered contrast between conventional ML and the \
3D-pretrained Uni-Mol family (Figure S1), {{crossover_statement}}. Tuned comparisons of D-MPNN and random forest \
report crossovers at roughly 500–2,000 training compounds {{cite:chen2025datascaling}}, and dataset size is known \
to govern when representation learning pays off {{cite:deng2023systematic}}. Under QSARena's single fixed \
configuration, {{crossover_comparison}}.

**Chemical-space shift.** {{snn_sentence}} {{strongest_sentence}} Message-passing models have \
been reported to generalise to unseen chemical space better than tree-based models {{cite:yang2019analyzing}}, \
and the {{n_resplit}} datasets re-split to scaffold splits between our two runs (main-text Section 3.10) offer a \
small natural experiment on identical chemistry (Figure S2b). {{natural_sentence}}

**Family × property grid.** Across {{n_grid_cells}} family × meta-feature correlations with Benjamini–Hochberg \
control (Table S10), {{grid_sentence}}.

**A family recommender.** A leave-one-dataset-out {{lodo_model}} used {{k_words}} meta-features: log10 training-set \
size, mean SNN, label asymmetry and task. It predicted the best family group (fusion, descriptor-based ML, \
Uni-Mol or Chemprop) with balanced accuracy {{lodo_bacc}} (95% CI {{lodo_bacc_ci}}). The majority-class baseline \
scored {{baseline_bacc}}, and label permutation scored {{lodo_perm_null}} (permutation p = {{lodo_perm_p}}; \
Figure S3). {{recommender_sentence}}

Predicting the winner's identity is a harsh test, because many datasets have several families within a few \
percent of each other. Algorithm selection is usually judged instead by regret, the performance given up by \
the recommended choice {{cite:olier2018metaqsar}}. Here regret is the gap of the recommended family, and the \
selector was designed and fixed before any result was seen. Each held-out dataset's selector was chosen by an \
inner leave-one-dataset-out loop among {{v2_n_candidates}} variants. These were nearest-dataset and per-family \
ridge models over size, similarity, chemistry, label-landscape and training-set cross-validation landmark \
features. {{v2_verdict}} The picked family was within 5% of the best on {{v2_nested_within5}} of datasets, \
against {{v2_sbs_within5}} for the single best family. The inner loop most often chose {{v2_most_picked}} \
({{v2_most_picked_share}} of held-out datasets; all variants in Table S11).

Diversity metrics depend on the fingerprint. Recomputed with ECFP6 at 4,096 bits, the fingerprint-dependent \
correlations kept their sign in {{sens_same_sign}} of cells and their significance classification in \
{{sens_sig_agree}}. These relationships are exploratory. They rest on one split and one seed per dataset and on \
n = {{n_datasets}} datasets, so they generate hypotheses for a multi-seed or learning-curve study; they do not \
establish selection rules.

![Figure S1. Training-set size and the family crossover.](manuscript_assets/figures/figureM1_size_crossover.png)

**Figure S1.** (a) Relative gap of each family's best model to the per-dataset best versus log10 training-set \
size, with Theil–Sen fits and dataset-bootstrap 95% bands (gaps capped at 50% for display). (b) Fitted gap curves \
for conventional ML and Uni-Mol; the dashed line and grey span mark the estimated crossover and its bootstrap CI \
when one exists. Single split and single seed per dataset: bands reflect dataset resampling, not seed variance.

![Figure S2. Chemical-space shift and dataset difficulty.](manuscript_assets/figures/figureM2_shift_difficulty.png)

**Figure S2.** (a) Achievable best held-out metric versus mean test-to-train nearest-neighbour Tanimoto (SNN), by \
task and split protocol. (b) Family gaps on the {{n_resplit}} datasets that were re-split from random or \
target-quartile splits to scaffold splits between the RTX 4060 and A100 runs (main-text Section 3.10); thin lines \
are datasets, thick lines are medians.

![Figure S3. Leave-one-dataset-out family recommender.](manuscript_assets/figures/figureM3_recommender.png)

**Figure S3.** (a) Depth-3 decision tree over four meta-features, fitted on all datasets for display. (b) \
Leave-one-dataset-out balanced accuracy of the tree and an L1-penalised multinomial logistic model against the \
majority-class baseline (grey), the label-permutation null (dotted) and chance (dashed), with dataset-bootstrap \
95% CIs. Exploratory: n = {{n_datasets}}.
"""

#: Main-text summary of the meta-analysis (Section 3.14).
SUMMARY = """\
### 3.14 When does each family win?

A dataset-property meta-analysis, a form of per-dataset algorithm selection from dataset meta-features \
{{cite:olier2018metaqsar}}, asked whether properties of the dataset alone predict which family comes close to the \
best. It was exploratory and, at n = {{n_datasets}}, inconclusive: {{n_grid_significant}} of {{n_grid_cells}} \
family × meta-feature correlations survived Benjamini–Hochberg control, a leave-one-dataset-out {{lodo_model}} \
predicted the best family group with balanced accuracy {{lodo_bacc}} against {{baseline_bacc}} for the majority \
class (permutation p = {{lodo_perm_p}}), and a pre-registered regret-based selector did not reliably beat always \
choosing the family with the best average record (permutation p = {{v2_perm_p}}). The per-dataset winner therefore \
remains an empirical question, which is the case for benchmarking many families on every dataset (Additional file 1, \
Note S4, Figures S1–S3 and Tables S9–S11).
"""

LIMITATION = """\
**Meta-analysis.** The meta-analysis (Section 3.14; Additional file 1, Note S4) relates dataset properties to \
family performance, but it inherits the single-split, single-seed design: individual winners are noisy, which is \
why it models the continuous gap and bootstraps over datasets, not seeds. With n = {{n_datasets}} datasets the \
meta-models are kept to four predictors, and the diversity and similarity features depend on the fingerprint used \
(the ECFP6 sensitivity check preserved the sign of {{sens_same_sign}} of the fingerprint-dependent correlations). \
How informative similarity-based domain metrics are itself varies with training-set diversity \
{{cite:sheridan2015relative}}.
"""

NESTED = """**Feature selection inside cross-validation.** Selecting features on the whole training split and then cross-validating models on the selected columns lets every held-out fold influence the selection, which inflates cross-validated scores. Selection was therefore refitted inside every cross-validation fold, on that fold's training rows only and with the method pinned to the one recorded for the full training split. These fold-specific selections produce both the cross-validated metrics of every model that uses the selected features and their out-of-fold ensemble predictions (§2.12); §3.4 reports the bias they remove. Where TabPFN's local fold refits did not fit the GPU, its cross-validated scores are withdrawn rather than reported from the leaky protocol.
"""

NESTED_RESULTS = """\
### 3.4 Nested feature selection removes most cross-validation optimism

Fitting feature selection once on the whole training split and then cross-validating on the selected columns lets \
every validation fold help choose its own features. A controlled comparison on {{nested_ct_datasets}} datasets, \
with identical features, models and folds and selection fitted either outside or inside the folds, isolated this \
effect: fitting selection outside the folds raised the overstatement of cross-validated over test RMSE by a median \
of {{nested_ct_per_model}}, and raised it in {{nested_ct_positive}} of {{nested_ct_pairs}} dataset-model pairs \
(sign test, {{nested_ct_p}}).

Across the benchmark, on datasets with predefined or scaffold test splits, nesting the selection (§2.5) cut the \
median overstatement of cross-validated over test scores from {{nested_outer}} to {{nested_nested}} across \
{{nested_pairs}} model-dataset pairs (lower in {{nested_share}} of them). That is the level shown by \
fixed-configuration models trained without feature selection on the same folds ({{cvleak_arm}}), so the optimism \
that remains reflects the gap between training-set folds and harder held-out splits rather than leakage (see \
Limitations). Test-set results are unchanged by construction: the deployed models and their selected features are \
the same. The correction matters in two places downstream. The cross-validation-selected protocol of §3.5 now rests \
on unbiased cross-validated scores, and the out-of-fold ensembles no longer over-weight members whose fold \
predictions benefited from the leak.
"""

CVLEAK = """**Residual cross-validation optimism.** With feature selection nested inside the folds (§2.5), cross-validated scores still exceed the corresponding test scores by a median of {{nested_nested}} on datasets with predefined or scaffold test splits (per-model medians {{cvleak_range}} across {{cvleak_n_models}} CV-scored models). Fixed-configuration models that select no features show {{cvleak_arm}} on the same folds, so the remainder reflects the gap between training-set folds and harder held-out splits rather than leakage. The cross-validation-selected protocol in §3.5 should be read with that margin in mind.
"""

BLOCKS = {"section_3_14": SECTION, "section_meta_summary": SUMMARY, "limitation_meta": LIMITATION,
          "limitation_cvleak": CVLEAK, "methods_nested_selection": NESTED,
          "results_nested_selection": NESTED_RESULTS}
#: Which LaTeX file holds each block: "main" = body.tex, "si" = additional_file_1.tex. The Markdown copy of every
#: block lives in manuscript.md (the Markdown supplement is part of that file).
BLOCK_TARGETS = {name: "main" for name in BLOCKS} | {"section_3_14": "si"}
#: \label for a block's subsection heading.
HEADING_LABELS = {"results_nested_selection": "sec:nestedcv", "section_meta_summary": "sec:meta",
                  "section_3_14": "note:meta"}

MD_MARKERS = ("<!-- META:{name} -->\n", "<!-- /META -->")
TEX_MARKERS = ("% META:{name}\n", "% /META")


def unresolved(template: str, macros: dict) -> list[str]:
    return sorted({m for m in MACRO.findall(template) if not m.startswith("cite:") and m not in macros})


#: Macro text written for the main text that refers to main-text figure numbers; the supplement renumbers them.
_SI_FIGURE_TEXT = {"Fig. 7": "Figure S1", "Fig. 8": "Figure S2", "Fig. 9": "Figure S3"}


def render(template: str, macros: dict, cite, si: bool = False) -> str:
    missing = unresolved(template, macros)
    if missing:
        raise KeyError(f"unresolved manuscript macros: {missing}")

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key.startswith("cite:"):
            return cite(key.split(":", 1)[1])
        value = str(macros[key])
        if si:
            for old, new in _SI_FIGURE_TEXT.items():
                value = value.replace(old, new)
        return value

    return MACRO.sub(replace, template)


def render_markdown(template: str, macros: dict, si: bool = False) -> str:
    return render(template, macros, lambda key: f"[{CITATIONS[key]}]", si=si)


_TEX_REPLACEMENTS = [
    ("%", r"\%"),
    ("&", r"\&"),
    ("≈", r"$\approx$"),
    ("ρ", r"$\rho$"),
    ("×", r"$\times$"),
    ("≥", r"$\geq$"),
    ("≤", r"$\leq$"),
    ("→", r"$\rightarrow$"),
    ("−", "--"),
    ("–", "--"),
    ("§", r"\S"),
]
_SECTION_LABELS = {
    "2.5": "sec:selection", "2.12": "sec:lineage", "3.2": "sec:nowinner", "3.4": "sec:nestedcv",
    "3.5": "sec:leaderboard", "3.10": "sec:hardware", "3.12": "sec:platforms", "3.14": "sec:meta",
}
_FIG_LABELS = {
    "figureM1_size_crossover": "fig:meta_size",
    "figureM2_shift_difficulty": "fig:meta_shift",
    "figureM3_recommender": "fig:meta_recommender",
}


def _tex_inline(text: str, si: bool = False) -> str:
    for old, new in _TEX_REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"Table S(\d+)", r"Table~S\1", text)
    text = re.sub(r"Figures? S(\d)", lambda m: m.group(0).replace(" ", "~"), text)
    if si:
        # A separate document: no \ref into the main text, so main-text sections and figures stay literal.
        text = text.replace("main-text Figure 6", "main-text Figure~6")
    else:
        text = text.replace("Fig. 6", r"Figure~\ref{fig:heatmap}")
        for number, label in _SECTION_LABELS.items():
            ref = rf"Section~\ref{{{label}}}"
            text = re.sub(rf"(\\S|Section ){re.escape(number)}(?!\d)", lambda _m, ref=ref: ref, text)
    return re.sub(r"(?<=[\s(])-(?=\d)", "$-$", text)


def render_latex(template: str, macros: dict, si: bool = False, heading_label: str = "") -> str:
    """Markdown template -> LaTeX paragraphs; images become figure environments with \\ref labels."""
    md = render(template, macros, lambda key: "{{CITE:" + key + "}}", si=si)
    out = []
    paragraphs = [p for p in md.split("\n\n") if p.strip()]
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()
        heading = re.match(r"^###\s+\d+\.\d+\s+(.*)$", para)
        note = re.match(r"^###\s+Supplementary Note (S\d+)\.\s+(.*)$", para)
        image = re.match(r"^!\[[^\]]*\]\(manuscript_assets/figures/([A-Za-z0-9_]+)\.png\)$", para)
        if heading:
            out.append(f"\\subsection{{{_tex_inline(heading.group(1), si)}}}\\label{{{heading_label}}}")
        elif note:
            out.append(f"\\section*{{Supplementary Note {note.group(1)}: {_tex_inline(note.group(2), si)}}}"
                       f"\\label{{{heading_label}}}")
        elif image:
            stem = image.group(1)
            caption = paragraphs[i + 1].strip() if i + 1 < len(paragraphs) else ""
            caption = re.sub(r"^\*\*Figure S?\d+\.\*\*\s*", "", caption)
            out.append(
                "\\begin{figure}[htbp]\n\\centering\n"
                f"\\includegraphics[width=\\textwidth]{{figures/{stem}.pdf}}\n"
                f"\\caption{{{_tex_inline(caption, si)}}}\\label{{{_FIG_LABELS[stem]}}}\n\\end{{figure}}"
            )
            i += 1
        else:
            out.append(_tex_inline(para, si))
        i += 1
    tex = "\n\n".join(out) + "\n"
    return re.sub(r"\{\{CITE:([A-Za-z0-9_]+)\}\}", r"~\\cite{\1}", tex).replace(" ~\\cite", "~\\cite")


def rendered_blocks(numbers: dict) -> dict[str, dict[str, str]]:
    macros = numbers["macros"]
    out = {}
    for name, template in BLOCKS.items():
        si = BLOCK_TARGETS[name] == "si"
        out[name] = {"md": render_markdown(template, macros, si=si),
                     "tex": render_latex(template, macros, si=si, heading_label=HEADING_LABELS.get(name, ""))}
    return out


def _block_pattern(markers: tuple[str, str], name: str) -> re.Pattern:
    start, end = markers
    return re.compile(re.escape(start.format(name=name)) + r"(?P<body>.*?)" + re.escape(end), re.DOTALL)


def extract_block(text: str, markers: tuple[str, str], name: str) -> str | None:
    match = _block_pattern(markers, name).search(text)
    return match.group("body") if match else None


def replace_block(text: str, markers: tuple[str, str], name: str, body: str) -> tuple[str, bool]:
    pattern = _block_pattern(markers, name)
    if not pattern.search(text):
        return text, False
    start, end = markers
    return pattern.sub(lambda _m: start.format(name=name) + body + end, text, count=1), True


def _targets(md_path: Path, tex_path: Path, si_tex_path: Path):
    """(path, markers, format key, block names) for every file that holds META blocks."""
    main = [n for n in BLOCKS if BLOCK_TARGETS[n] == "main"]
    si = [n for n in BLOCKS if BLOCK_TARGETS[n] == "si"]
    return ((md_path, MD_MARKERS, "md", list(BLOCKS)), (tex_path, TEX_MARKERS, "tex", main),
            (si_tex_path, TEX_MARKERS, "tex", si))


def sync_manuscript(numbers: dict, md_path: Path = MANUSCRIPT_MD, tex_path: Path = BODY_TEX,
                    si_tex_path: Path = SI_TEX) -> list[str]:
    """Rewrite every META block in both manuscript formats; report blocks whose markers are missing."""
    report = []
    blocks = rendered_blocks(numbers)
    for path, markers, key, names in _targets(md_path, tex_path, si_tex_path):
        if not Path(path).exists():
            report.append(f"missing {path}")
            continue
        text = Path(path).read_text(encoding="utf-8")
        for name in names:
            text, found = replace_block(text, markers, name, blocks[name][key])
            report.append(f"{'synced' if found else 'NO MARKERS for'} {name} in {Path(path).name}")
        Path(path).write_text(text, encoding="utf-8")
    return report


def check_manuscript(numbers: dict, md_path: Path = MANUSCRIPT_MD, tex_path: Path = BODY_TEX,
                     si_tex_path: Path = SI_TEX) -> list[str]:
    """Problems found: missing markers, or blocks that differ from a fresh render of meta_numbers.json."""
    problems = []
    blocks = rendered_blocks(numbers)
    for path, markers, key, names in _targets(md_path, tex_path, si_tex_path):
        text = Path(path).read_text(encoding="utf-8")
        for name in names:
            body = extract_block(text, markers, name)
            if body is None:
                problems.append(f"{Path(path).name}: META block {name} missing")
            elif body != blocks[name][key]:
                problems.append(f"{Path(path).name}: META block {name} drifted from meta_numbers.json")
    return problems
