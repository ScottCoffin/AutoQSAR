"""Manuscript text for the meta-analysis, rendered from ``meta_numbers.json`` macros.

Every quantitative claim is a ``{{macro}}`` placeholder filled from ``meta_numbers.json["macros"]``;
conditional wording (e.g. whether a trend narrows) is decided in ``pipeline.build_numbers`` and
arrives as a macro too. The rendered text lives between marker comments in both manuscript formats:

- ``manuscript.md``:  ``<!-- META:<block> -->`` ... ``<!-- /META -->``
- ``submission/body.tex``: ``% META:<block>`` ... ``% /META``

Never hand-edit inside those markers. ``verify_manuscript_numbers.py`` re-renders the blocks from
``meta_numbers.json`` and fails if the manuscript drifts from the numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

from qsarena.meta_analysis.io import REPO_ROOT

MANUSCRIPT_MD = REPO_ROOT / "manuscript.md"
BODY_TEX = REPO_ROOT / "submission" / "body.tex"
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

SECTION = """\
### 3.14 When does each family win? A dataset-property meta-analysis

Section 3.2 found that no model family dominates. Here we ask when each family comes close to the best, using \
properties of the dataset alone. This is per-dataset algorithm selection from dataset meta-features, the \
approach Meta-QSAR applied to thousands of QSAR problems {{cite:olier2018metaqsar}}, and it addresses the \
observation that the best ADMET model and representation are strongly dataset-dependent \
{{cite:kamuntavicius2025benchmarking}}. For each of the {{n_datasets}} datasets we computed meta-features on the \
exact train/test partition used in the benchmark, verified against the recorded split hashes (Table S9). They are: \
training-set size, label imbalance or skew, Bemis–Murcko scaffold diversity, internal fingerprint diversity, and \
train-to-test similarity. Similarity is summarised by SNN, the Tanimoto similarity of each test molecule to its \
nearest training molecule, which is known to track prediction error {{cite:sheridan2004similarity}}. Winners are \
single-split, single-seed outcomes, so the outcome modelled is the continuous relative gap of Fig. 6, never the \
identity of the winner. Every interval is a 95% percentile bootstrap over datasets.

**Training-set size.** {{size_sentence}} For the pre-registered contrast between conventional ML and the \
3D-pretrained Uni-Mol family (Fig. 7), {{crossover_statement}}. Tuned comparisons of D-MPNN and random forest \
report crossovers at roughly 500–2,000 training compounds {{cite:chen2025datascaling}}, and dataset size is known \
to govern when representation learning pays off {{cite:deng2023systematic}}. Under QSARena's single fixed \
configuration, {{crossover_comparison}}.

**Chemical-space shift.** {{snn_sentence}} {{strongest_sentence}} Message-passing models have \
been reported to generalise to unseen chemical space better than tree-based models {{cite:yang2019analyzing}}, \
and the {{n_resplit}} datasets re-split to scaffold splits between our two runs (§3.10) offer a small natural \
experiment on identical chemistry (Fig. 8b). {{natural_sentence}}

**Family × property grid.** Across {{n_grid_cells}} family × meta-feature correlations with Benjamini–Hochberg \
control (Table S10), {{grid_sentence}}.

**A family recommender.** A leave-one-dataset-out {{lodo_model}} used {{k_words}} meta-features: log10 training-set \
size, mean SNN, label asymmetry and task. It predicted the best family group (fusion, descriptor-based ML, \
Uni-Mol or Chemprop) with balanced accuracy {{lodo_bacc}} (95% CI {{lodo_bacc_ci}}). The majority-class baseline \
scored {{baseline_bacc}}, and label permutation scored {{lodo_perm_null}} (permutation p = {{lodo_perm_p}}; Fig. 9). \
{{recommender_sentence}}

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

![Figure 7. Training-set size and the family crossover.](manuscript_assets/figures/figureM1_size_crossover.png)

**Figure 7.** (a) Relative gap of each family's best model to the per-dataset best versus log10 training-set \
size, with Theil–Sen fits and dataset-bootstrap 95% bands (gaps capped at 50% for display). (b) Fitted gap curves \
for conventional ML and Uni-Mol; the dashed line and grey span mark the estimated crossover and its bootstrap CI \
when one exists. Single split and single seed per dataset: bands reflect dataset resampling, not seed variance.

![Figure 8. Chemical-space shift and dataset difficulty.](manuscript_assets/figures/figureM2_shift_difficulty.png)

**Figure 8.** (a) Achievable best held-out metric versus mean test-to-train nearest-neighbour Tanimoto (SNN), by \
task and split protocol. (b) Family gaps on the {{n_resplit}} datasets that were re-split from random or \
target-quartile splits to scaffold splits between the RTX 4060 and A100 runs (§3.10); thin lines are datasets, \
thick lines are medians.

![Figure 9. Leave-one-dataset-out family recommender.](manuscript_assets/figures/figureM3_recommender.png)

**Figure 9.** (a) Depth-3 decision tree over four meta-features, fitted on all datasets for display. (b) \
Leave-one-dataset-out balanced accuracy of the tree and an L1-penalised multinomial logistic model against the \
majority-class baseline (grey), the label-permutation null (dotted) and chance (dashed), with dataset-bootstrap \
95% CIs. Exploratory: n = {{n_datasets}}.
"""

LIMITATION = """\
**Meta-analysis.** Section 3.14 relates dataset properties to family performance, but it inherits the \
single-split, single-seed design: individual winners are noisy, which is why it models the continuous gap and \
bootstraps over datasets, not seeds. With n = {{n_datasets}} datasets the meta-models are kept to four predictors, \
and the diversity and similarity features depend on the fingerprint used (the ECFP6 sensitivity check preserved \
the sign of {{sens_same_sign}} of the fingerprint-dependent correlations). How informative similarity-based \
domain metrics are itself varies with training-set diversity {{cite:sheridan2015relative}}.
"""

CVLEAK = """**Model-selection cross-validation is optimistic.** The runner fits the feature selector once on the full training split and then cross-validates each model on the selected features, so every held-out CV fold has already influenced which features were kept. On datasets with predefined or scaffold test splits, CV scores overstate the corresponding test scores by a median of {{cvleak_median}} across {{cvleak_n_models}} CV-scored models (per-model medians {{cvleak_range}}; largest for {{cvleak_top}}), against {{cvleak_arm}} for fixed-configuration tree models trained on unselected features with the same folds. Test-set results and leaderboard placements are unaffected, because selection never sees test data, but the cross-validation-selected model and the CV-to-test gap in §3.4 rest on these optimistic scores. Nesting feature selection inside each CV fold would remove the bias and is left to future work.
"""

BLOCKS = {"section_3_14": SECTION, "limitation_meta": LIMITATION, "limitation_cvleak": CVLEAK}

MD_MARKERS = ("<!-- META:{name} -->\n", "<!-- /META -->")
TEX_MARKERS = ("% META:{name}\n", "% /META")


def unresolved(template: str, macros: dict) -> list[str]:
    return sorted({m for m in MACRO.findall(template) if not m.startswith("cite:") and m not in macros})


def render(template: str, macros: dict, cite) -> str:
    missing = unresolved(template, macros)
    if missing:
        raise KeyError(f"unresolved manuscript macros: {missing}")

    def replace(match: re.Match) -> str:
        key = match.group(1)
        if key.startswith("cite:"):
            return cite(key.split(":", 1)[1])
        return str(macros[key])

    return MACRO.sub(replace, template)


def render_markdown(template: str, macros: dict) -> str:
    return render(template, macros, lambda key: f"[{CITATIONS[key]}]")


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
_SECTION_LABELS = {"3.2": "sec:nowinner", "3.10": "sec:hardware", "3.14": "sec:meta"}
_FIG_LABELS = {
    "figureM1_size_crossover": "fig:meta_size",
    "figureM2_shift_difficulty": "fig:meta_shift",
    "figureM3_recommender": "fig:meta_recommender",
}


def _tex_inline(text: str) -> str:
    for old, new in _TEX_REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(
        r"(?<![\\\w])Fig\. ([789])",
        lambda m: f"Figure~\\ref{{{list(_FIG_LABELS.values())[int(m.group(1)) - 7]}}}",
        text,
    )
    text = re.sub(r"Figure~\\ref\{([a-z_:]+)\}([ab])", r"Figure~\\ref{\1}\2", text)
    text = re.sub(r"Table S(\d+)", r"Table~S\1", text)
    text = text.replace("Fig. 6", r"Figure~\ref{fig:heatmap}")
    for number, label in _SECTION_LABELS.items():
        ref = rf"Section~\ref{{{label}}}"
        text = re.sub(rf"(\\S|Section ){re.escape(number)}(?!\d)", lambda _m, ref=ref: ref, text)
    return re.sub(r"(?<=[\s(])-(?=\d)", "$-$", text)


def render_latex(template: str, macros: dict) -> str:
    """Markdown template -> LaTeX paragraphs; images become figure environments with \\ref labels."""
    md = render(template, macros, lambda key: "{{CITE:" + key + "}}")
    out = []
    paragraphs = [p for p in md.split("\n\n") if p.strip()]
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i].strip()
        heading = re.match(r"^###\s+\d+\.\d+\s+(.*)$", para)
        image = re.match(r"^!\[[^\]]*\]\(manuscript_assets/figures/([A-Za-z0-9_]+)\.png\)$", para)
        if heading:
            out.append(f"\\subsection{{{_tex_inline(heading.group(1))}}}\\label{{sec:meta}}")
        elif image:
            stem = image.group(1)
            caption = paragraphs[i + 1].strip() if i + 1 < len(paragraphs) else ""
            caption = re.sub(r"^\*\*Figure \d+\.\*\*\s*", "", caption)
            out.append(
                "\\begin{figure}[htbp]\n\\centering\n"
                f"\\includegraphics[width=\\textwidth]{{figures/{stem}.pdf}}\n"
                f"\\caption{{{_tex_inline(caption)}}}\\label{{{_FIG_LABELS[stem]}}}\n\\end{{figure}}"
            )
            i += 1
        else:
            out.append(_tex_inline(para))
        i += 1
    tex = "\n\n".join(out) + "\n"
    return re.sub(r"\{\{CITE:([A-Za-z0-9_]+)\}\}", r"~\\cite{\1}", tex).replace(" ~\\cite", "~\\cite")


def rendered_blocks(numbers: dict) -> dict[str, dict[str, str]]:
    macros = numbers["macros"]
    return {
        name: {"md": render_markdown(template, macros), "tex": render_latex(template, macros)}
        for name, template in BLOCKS.items()
    }


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


def sync_manuscript(numbers: dict, md_path: Path = MANUSCRIPT_MD, tex_path: Path = BODY_TEX) -> list[str]:
    """Rewrite every META block in both manuscript formats; report blocks whose markers are missing."""
    report = []
    blocks = rendered_blocks(numbers)
    for path, markers, key in ((md_path, MD_MARKERS, "md"), (tex_path, TEX_MARKERS, "tex")):
        if not Path(path).exists():
            report.append(f"missing {path}")
            continue
        text = Path(path).read_text(encoding="utf-8")
        for name, bodies in blocks.items():
            text, found = replace_block(text, markers, name, bodies[key])
            report.append(f"{'synced' if found else 'NO MARKERS for'} {name} in {Path(path).name}")
        Path(path).write_text(text, encoding="utf-8")
    return report


def check_manuscript(numbers: dict, md_path: Path = MANUSCRIPT_MD, tex_path: Path = BODY_TEX) -> list[str]:
    """Problems found: missing markers, or blocks that differ from a fresh render of meta_numbers.json."""
    problems = []
    blocks = rendered_blocks(numbers)
    for path, markers, key in ((md_path, MD_MARKERS, "md"), (tex_path, TEX_MARKERS, "tex")):
        text = Path(path).read_text(encoding="utf-8")
        for name, bodies in blocks.items():
            body = extract_block(text, markers, name)
            if body is None:
                problems.append(f"{Path(path).name}: META block {name} missing")
            elif body != bodies[key]:
                problems.append(f"{Path(path).name}: META block {name} drifted from meta_numbers.json")
    return problems
