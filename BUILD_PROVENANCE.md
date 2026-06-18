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

## pip freeze inside image

<!-- populated by `make image` — docker run --rm autoqsar:<commit> pip freeze -->

## .sif SHA-256

<!-- populated by `make sif` — sha256sum autoqsar.sif -->
