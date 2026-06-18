# Running AutoQSAR on Jetstream2 (OpenStack Cloud)

This directory contains the Jetstream2-specific execution path for AutoQSAR.
The v1 Slurm scripts for NSF ACCESS HPC clusters are preserved under `hpc/` and
are unaffected by the additions here.

---

## Flavor → SU rate table

| Flavor | vCPU | RAM | GPU | SU/hr | Recommended for |
|---|---|---|---|---|---|
| `g3.large`  | 10 | 60 GB | 50% A100 (20 GB VRAM) | **32** | MapLight+GNN, TabPFN, Uni-Mol2, Chemprop |
| `m3.medium` |  8 | 30 GB | — | **8**  | Conventional ML, CFA fusion, conformer pre-gen |
| `m3.large`  | 16 | 60 GB | — | **16** | Heavy CPU workloads |

> SU/hr rates as of mid-2026. Verify at https://jetstream-cloud.org/allotments/su_calculator.html

---

## Persistent volume layout

All data and outputs should live on an attached Cinder volume (not the ephemeral
root disk) so they survive shelve/unshelve cycles.

```
/vol/autoqsar/
├── autoqsar_in/              # bind-mounted as /in inside container
│   ├── data/                 # TDC-22 dataset files (.tab, .csv)
│   ├── model_cache/          # pre-trained MapLight feature stores
│   ├── .cache/               # vendored PyTDC tarball
│   └── conformer_cache/      # SQLite conformer cache (autoqsar/conformers.py)
├── autoqsar_out/             # bind-mounted as /out; task outputs land here
│   └── <dataset>/seed_<N>/  # one directory per (dataset, seed) task
└── queue_state.json          # orchestrator checkpoint (do NOT delete while running)
```

Recommended setup after provisioning:
```bash
sudo mkfs.ext4 /dev/sdb
sudo mkdir -p /vol/autoqsar
sudo mount /dev/sdb /vol/autoqsar
sudo chown $USER:$USER /vol/autoqsar
mkdir -p /vol/autoqsar/{autoqsar_in/data,autoqsar_in/model_cache,autoqsar_in/.cache,autoqsar_in/conformer_cache,autoqsar_out}
```

---

## Workflow routing: CPU vs GPU

| Workflow family | Winner datasets | Recommended instance |
|---|---|---|
| MapLight+GNN | `tdc_ppbr_az`, `tdc_vdss_lombardo`, `tdc_clearance_microsome_az`, `tdc_clearance_hepatocyte_az` | g3.large |
| TabPFN | `tdc_half_life_obach` | g3.large (TabPFN benefits from GPU when N>500) |
| Uni-Mol2 84m (§B) | any dataset with `--run-unimol-v2` | g3.large (6 GB VRAM) |
| Conventional ML, CFA fusion, XGBoost | remaining 17 datasets | m3.medium |
| Conformer pre-generation | run before GPU tasks | m3.medium |

Manifests:
- `js2/manifests/gpu_tasks.tsv` — 25 GPU tasks (5 datasets × 5 seeds)
- `js2/manifests/cpu_tasks.tsv` — 85 CPU tasks (17 datasets × 5 seeds)

---

## Quick-start: GPU queue

```bash
# 1. Provision g3.large on Jetstream2 Horizon or CLI, attach volume, boot
# 2. SSH in and run preflight
bash js2/preflight.sh --gpu-required

# 3. Transfer image and data (from local machine)
scp autoqsar.sif user@<JS2_IP>:~/autoqsar/
rsync -av data/ user@<JS2_IP>:/vol/autoqsar/autoqsar_in/data/
rsync -av model_cache/ user@<JS2_IP>:/vol/autoqsar/autoqsar_in/model_cache/
rsync -av .cache/ user@<JS2_IP>:/vol/autoqsar/autoqsar_in/.cache/

# 4. Run the GPU queue (on the Jetstream2 VM)
export INSTANCE_ID="<your-openstack-instance-uuid>"
python js2/run_queue.py \
    --manifest js2/manifests/gpu_tasks.tsv \
    --sif ~/autoqsar/autoqsar.sif \
    --input-dir /vol/autoqsar/autoqsar_in \
    --output-dir /vol/autoqsar/autoqsar_out \
    --state /vol/autoqsar/queue_state_gpu.json \
    --instance-flavor g3.large \
    --su-budget 200 \
    --precision tf32_bf16
```

## Quick-start: CPU queue

```bash
# Provision m3.medium instance
bash js2/preflight.sh   # no --gpu-required

python js2/run_queue.py \
    --manifest js2/manifests/cpu_tasks.tsv \
    --sif ~/autoqsar/autoqsar.sif \
    --input-dir /vol/autoqsar/autoqsar_in \
    --output-dir /vol/autoqsar/autoqsar_out \
    --state /vol/autoqsar/queue_state_cpu.json \
    --instance-flavor m3.medium \
    --su-budget 100 \
    --precision fp32
```

---

## Conformer pre-generation (standalone CPU mode)

Generate and cache 3D conformers on m3.medium before sending GPU tasks, so
the GPU instance spends no time on CPU conformer geometry:

```bash
# On m3.medium (or any CPU host with RDKit installed)
for DATASET in tdc_ppbr_az tdc_vdss_lombardo tdc_clearance_microsome_az \
               tdc_clearance_hepatocyte_az tdc_half_life_obach; do
    apptainer run autoqsar.sif \
        --generate-conformers-only \
        --dataset $DATASET \
        --input-dir /vol/autoqsar/autoqsar_in \
        --conformer-cache /vol/autoqsar/autoqsar_in/conformer_cache \
        --conformer-seed 42
done
```

The cache is a SQLite database at `/vol/autoqsar/autoqsar_in/conformer_cache/conformer_cache.db`.
GPU tasks read from it automatically when `--conformer-cache` is set.

---

## Shelve / unshelve cycle

The orchestrator tracks SU budget and auto-shelves when exhausted:

```bash
# Shelve (stops billing):
openstack server shelve <INSTANCE_ID>

# Unshelve (resumes instance from saved state):
openstack server unshelve <INSTANCE_ID>

# Resume queue (state file persists on the attached volume):
python js2/run_queue.py --manifest ... --state /vol/autoqsar/queue_state_gpu.json ...
```

The `queue_state.json` file records which tasks are `done`, `error`, or interrupted.
On resume, only `pending` and `running` (interrupted) tasks are re-queued.

---

## Retrieving results

```bash
# From local machine
rsync -av user@<JS2_IP>:/vol/autoqsar/autoqsar_out/ ./benchmark_results/js2_multiseed/
```

Each task writes to `autoqsar_out/<dataset>/seed_<N>/` with the same output
structure as the Slurm HPC path (see `CONTAINER.md §8`).

---

## v1 Slurm scripts

The Slurm submission scripts are unchanged under `hpc/`:
- `hpc/submit_multiseed.sbatch` — GPU array
- `hpc/submit_multiseed_cpu.sbatch` — CPU array
- `hpc/task_manifest.tsv` — full 110-task manifest (22 datasets × 5 seeds)

These work on any NSF ACCESS cluster (Delta, Bridges-2, Expanse, Stampede3)
and are independent of the Jetstream2 path.
