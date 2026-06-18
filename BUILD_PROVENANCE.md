# AutoQSAR Build Provenance

This file is a stub that is auto-populated by `make image` and `make sif`.
Fields below will be filled by the build process.

## Build metadata

- git_commit: <!-- populated by make image -->
- git_tag: <!-- populated by make image -->
- build_date: <!-- populated by make image -->
- image_tag: <!-- populated by make image -->
- sif_sha256: <!-- populated by make sif -->
- sif_path: <!-- populated by make sif -->

## Audit-time state

- Audit-time commit: 46bcd3d (reorg and draft manuscript)
- Audit-time branch: feat/containerize
- Audit-time tree-clean: true (verified 2026-06-18)
- git_is_dirty: false at image build time (enforced by Makefile; see audit_report.md §1.4)

## Dependency gaps resolved in this containerization

1. `lightgbm==4.5.0` added — was imported in code (line 284) but missing from all
   requirements and environment files. Now pinned in environment/requirements-lock.txt.
2. `tabpfn-client` and `tabpfn-common-utils` pinned to 0.2.8 / 0.2.20 (were `>=` in cuda req).
3. Windows-only packages (`pywin32`, `pywinpty`, `win32-setctime`) excluded.
4. `numpy` unified to `2.2.6` (cuda req had floating `>=2.0,<2.5`).

## Update 2 additions (feat/update2-precision-unimol2-js2)

### §A — Precision config

- `precision_mode`: `tf32_bf16` or `fp32` (set via `--precision` CLI flag / `AUTOQSAR_PRECISION` env var)
- `tf32_enabled`: true when `precision_mode == tf32_bf16` and CUDA available
- `bf16_amp_available`: true when `precision_mode == tf32_bf16` and CUDA available
- `host_driver_version`: captured via `nvidia-smi` at run time
- `gpu_identity`: GPU name / UUID from `nvidia-smi`
- `precision_nondeterminism_note`: "TF32 introduces minor numerical non-determinism vs fp32; use --precision fp32 for the reproducibility reference run" (logged when mode == tf32_bf16)

### §B — Uni-Mol2 V2

- `unimolv2_size`: model checkpoint size (`84m` default)
- `unimolv2_checkpoint_sha256`: SHA-256 of `model_cache/unimolv2_checkpoints/<size>/checkpoint.pt`; operator must fill `autoqsar/unimolv2.py:UNIMOLV2_CHECKPOINT_SHA256` after first download
- `unimolv2_3d_input_flag`: always `true` (Uni-Mol2 consumes ETKDGv3+MMFF 3D geometry)
- `conformer_method`: `ETKDGv3+MMFF`
- `conformer_seed`: default `42` (separate from model seed; same geometry across all 5 model seeds)
- `conformer_cache_hash`: SHA-256 over all `(smiles, seed, mol_binary)` rows in `conformer_cache.db`, recorded at end of run

### §C — Jetstream2 path

- `js2/run_queue.py` tracks `elapsed_seconds` per task; SU burn = `elapsed_hours × su_rate`
- State file `queue_state.json` persists to the attached Cinder volume, surviving shelve cycles

### Unicore compilation

- `unicore` (DeepModeling backbone) is compiled from source in the Dockerfile devel stage
- Pin: `UNICORE_GIT_SHA=e04f7ef73d7685cf0fd1090de87a1b0900ef6f01` (update in Dockerfile ARG after confirming)
- `unimol-tools==1.0.1` (V2 support; was 0.1.5 in v1)

## pip freeze inside image

<!-- populated by `make image` — docker run --rm autoqsar:<commit> pip freeze -->

## .sif SHA-256

<!-- populated by `make sif` — sha256sum autoqsar.sif -->
