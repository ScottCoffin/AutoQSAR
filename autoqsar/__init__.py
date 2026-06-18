"""
autoqsar — shared utilities for the AutoQSAR HPC containerization layer.

Modules:
  precision   — TF32/BF16 precision startup hook for all torch-based workflows
  conformers  — RDKit ETKDGv3+MMFF 3D conformer generation with persistent cache
  unimolv2    — Uni-Mol2 V2 workflow wrapper (84M default, OOM retry, bf16 AMP)
"""
