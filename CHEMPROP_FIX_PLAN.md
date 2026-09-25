# Agent scope plan — repair the Chemprop v2 backend

**Status:** open, blocking the paper's ADMET-AI parity claim.
**Priority:** highest open engineering item. Second only to the five-seed replication scientifically,
and unlike that one this is cheap.

---

## 1. Finding: the two runs did *not* fail the same way

The question asked was whether the A100 alone failed on the ADMET-AI-equivalent architecture, or
whether the RTX run also failed at comparable rates. They failed at **different rates, for entirely
different reasons, and the two failure modes are mutually exclusive by platform.**

| | A100 (Jetstream2, Linux) — canonical | RTX 4060 (Windows) |
|---|---|---|
| Chemprop variants configured | 5 | 2 |
| ADMET-AI-equivalent (D-MPNN + RDKit2D) configured | **Yes** | **No** |
| Valid on | **6 / 44 datasets (14%)** | 23 / 45 datasets (51%) |
| Per-variant failure rate | 86.4%, *identical across all five variants* | 48.9% and 53.3% |
| Dominant error | `'ascii' codec can't decode byte 0xe2 … ordinal not in range(128)` (728 rows) | `Chemprop command failed during training (exit=1)` (136 rows) |
| `ascii` decode errors present | 728 | **0** |
| Chemprop training failures present | 2 | 136+ |

Two distinct bugs, therefore:

- **Bug A (A100, dominant).** Not a training failure at all. A `UnicodeDecodeError` raised *inside*
  `subprocess.run` before the training result was ever inspected.
- **Bug B (RTX).** Genuine Chemprop training/prediction failures — non-zero exit codes and
  prediction-length mismatches.

The identical 86.4% rate across all five A100 variants is the tell: a model-quality problem would not
fail five different architectures at exactly the same rate. An I/O bug in the shared wrapper would.

## 2. Root cause of Bug A (confirmed, one line)

`portable_colab_qsar_bundle/run_qsarena_benchmarks.py:5729`, inside `_run_chemprop_command`:

```python
result = subprocess.run(cmd, capture_output=True, text=True)   # no encoding=
```

`text=True` without an explicit `encoding` decodes the child's stdout/stderr using
`locale.getpreferredencoding(False)`. On the Jetstream2 instance that resolved to **ANSI_X3.4-1968
(ASCII)** — the usual result of a `C`/`POSIX` locale in a container or non-login SSH session.
Chemprop v2 emits UTF-8 in its progress output (`0xe2` is the lead byte of `—`, `─`, `✓`, `→`),
so the decode raises and the whole call fails.

This explains every observation:

- Why Windows was unaffected: cp1252 maps `0xe2` to `â` without error. **Bug A cannot occur on Windows.**
- Why the byte offset varies (`position 146…3460`): it depends on how much output preceded the first
  non-ASCII character.
- Why exactly 6 datasets survived: short/plain output that happened to contain no non-ASCII byte.
  Those 6 are `chemml_cep_homo`, `chemml_organic_density`, `tdc_carcinogens_lagunin`, `tdc_dili`,
  `tdc_hia_hou`, `tdc_skin_reaction`.
- Why `_resolve_chemprop_command` (line 5713, same defect) mostly worked: `--help` output is ASCII.

## 3. Scope for the agent

### Must do

1. **Fix the decode.** Add `encoding="utf-8", errors="replace"` to the `subprocess.run` calls at
   lines 5713 and 5729. Audit the file for any other `text=True` / `capture_output=True` call with no
   explicit encoding (there are several: lines 522, 557, 4075) and fix them the same way.
2. **Make the failure legible.** A `UnicodeDecodeError` should never be recorded as if it were a
   model result. Wrap the call so that an exception raised while *reading* the subprocess is reported
   distinctly from a non-zero exit, e.g. `"chemprop harness error (decode): …"` versus
   `"chemprop training failed (exit=N)"`. The benchmark's `error` column currently conflates them,
   which is what hid this for a whole run.
3. **Add a regression test** that runs the wrapper against a stub emitting UTF-8 box-drawing
   characters on stdout under `LC_ALL=C`, and asserts it succeeds. This must fail before the fix and
   pass after.
4. **Prove the fix on real data.** Re-run Chemprop for a small sample — suggest 6 datasets spanning
   both task types and sizes, e.g. `tdc_caco2_wang`, `tdc_bbb_martins`, `tdc_ld50_zhu`,
   `tdc_ppbr_az`, `tdc_ames`, `tdc_herg` — with `--only-model-names` scoping if available, into a
   scratch `--output-dir`. **Do not touch `benchmark_results/autoqsar_benchmark_20260623_153839/`**;
   it is the canonical artifact set the manuscript regenerates from. Report valid-vs-attempted before
   and after.

### Should do

5. **Diagnose Bug B separately.** Collect the full stderr tails behind the 136 Windows `exit=1`
   failures in `benchmark_results/benchmark_name_date/` and classify them (dependency, CUDA/DLL,
   data-shape, OOM). Do not attempt a Windows fix in the same change; report findings.
6. **Investigate the prediction-length mismatches** (`got 531 rows, expected 532`;
   `got 0 rows, expected 904`) — these look like a row-alignment or silent-drop bug in
   `_align_predictions_by_smiles_occurrence`, independent of platform.

### Must not do

- Do not modify `manuscript.md`, `submission/`, `manuscript_assets/`, or any existing
  `benchmark_results/` directory.
- Do not re-run the full 44-dataset benchmark (≈112 h on an A100).
- Do not change Chemprop hyperparameters while fixing the harness; the point is to isolate the I/O
  bug from model behaviour.

## 4. Why this matters for the paper

The `Chemprop v2 (D-MPNN + RDKit2D)` variant is our ADMET-AI-equivalent architecture. It is currently
valid on 6 of 44 datasets, which is why the manuscript states that parity with ADMET-AI is **untested,
not lost**, and why the title's "match pretrained molecular models" claim is explicitly scoped to
Uni-Mol rather than ADMET-AI.

If the fix restores Chemprop coverage across the suite, three things become possible:

1. A genuine like-for-like ADMET-AI proxy, which is the cleanest route to earning the parity claim.
2. Table 3's Chemprop row becomes comparable with the others instead of being the one row we warn
   readers not to compare.
3. The largest stated threat to the completeness of Figure 2 (§3.9) is removed.

Every one of those is a direct response to a reviewer concern. Against that, the fix itself is
plausibly a one-line change plus a test.

## 5. Acceptance criteria

- [ ] All `subprocess.run` calls in the runner specify an explicit encoding.
- [ ] Harness/decode errors are recorded distinctly from model failures.
- [ ] A regression test reproduces the ASCII-locale failure and passes after the fix.
- [ ] A scratch re-run shows Chemprop valid on substantially more than 14% of the sampled datasets,
      with before/after numbers reported.
- [ ] Bug B classified, with a recommendation but no speculative Windows fix.
- [ ] No canonical artifact, manuscript or submission file modified.
