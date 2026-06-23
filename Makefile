# AutoQSAR Makefile
#
# Targets:
#   make env      Install pinned CPU environment in a local venv
#   make image    Build the Docker image (requires clean git tree + Docker)
#   make sif      Convert Docker image to Apptainer .sif
#   make smoke    Run CPU smoke test inside the container
#   make manifests Generate GPU and CPU sub-manifests from task_manifest.tsv

PYTHON        ?= python3.11
VENV_DIR      ?= .venv
IMAGE_NAME    ?= autoqsar
IMAGE_TAG     ?= latest
SIF_PATH      ?= autoqsar.sif
SMOKE_DATASET ?= tdc_herg
SMOKE_SEED    ?= 1

# Derived vars
GIT_COMMIT    := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
GIT_TAG       := $(shell git describe --tags --exact-match 2>/dev/null || echo untagged)
BUILD_DATE    := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
TREE_DIRTY    := $(shell git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

CONDA_ENV     ?= autoqsar-py311
BENCHMARK_OUTPUT_DIR ?=

.PHONY: env image sif smoke manifests clean help benchmark-a100

help:
	@echo "AutoQSAR container build targets:"
	@echo "  make env             Install pinned CPU venv (for local dev/test)"
	@echo "  make image           Build Docker image (requires clean git tree)"
	@echo "  make sif             Convert Docker image to Apptainer .sif"
	@echo "  make smoke           Run CPU smoke test inside container"
	@echo "  make manifests       Split task_manifest.tsv into GPU/CPU sub-manifests"
	@echo "  make benchmark-a100  Run (or resume) the full A100 benchmark"
	@echo "    BENCHMARK_OUTPUT_DIR=/path/to/run  resume a specific run directory"
	@echo "  make clean           Remove local build artifacts"

# ── env: install pinned CPU venv ─────────────────────────────────────────────
env:
	@echo "Creating venv at $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --no-cache-dir --upgrade pip setuptools wheel
	$(VENV_DIR)/bin/pip install --no-cache-dir \
		-r environment/requirements-lock.txt
	@echo "Done. Activate with: source $(VENV_DIR)/bin/activate"

# ── image: build Docker image (enforces clean tree) ──────────────────────────
image:
	@if [ "$(TREE_DIRTY)" != "0" ]; then \
		echo "ERROR: Working tree is dirty ($(TREE_DIRTY) modified files)."; \
		echo "       Commit or stash all changes before building a release image."; \
		echo "       A dirty-tree build would not be reproducible."; \
		exit 1; \
	fi
	@echo "Building Docker image $(IMAGE_NAME):$(GIT_COMMIT)..."
	@# Tag the commit for provenance (non-destructive; skip if tag already exists)
	@git tag autoqsar-v$(shell date -u +%Y%m%d) 2>/dev/null || true
	docker build \
		--file containers/Dockerfile \
		--tag $(IMAGE_NAME):$(GIT_COMMIT) \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--build-arg GIT_COMMIT="$(GIT_COMMIT)" \
		--build-arg GIT_TAG="$(GIT_TAG)" \
		--build-arg BUILD_DATE="$(BUILD_DATE)" \
		.
	@echo "Image built: $(IMAGE_NAME):$(GIT_COMMIT)"
	@# Record provenance
	@echo "## Docker image provenance" >> BUILD_PROVENANCE.md
	@echo "" >> BUILD_PROVENANCE.md
	@echo "- git_commit: $(GIT_COMMIT)" >> BUILD_PROVENANCE.md
	@echo "- git_tag: $(GIT_TAG)" >> BUILD_PROVENANCE.md
	@echo "- build_date: $(BUILD_DATE)" >> BUILD_PROVENANCE.md
	@echo "- image_tag: $(IMAGE_NAME):$(GIT_COMMIT)" >> BUILD_PROVENANCE.md
	@echo "" >> BUILD_PROVENANCE.md
	@echo "## pip freeze inside image" >> BUILD_PROVENANCE.md
	@echo '```' >> BUILD_PROVENANCE.md
	docker run --rm $(IMAGE_NAME):$(GIT_COMMIT) pip freeze >> BUILD_PROVENANCE.md
	@echo '```' >> BUILD_PROVENANCE.md

# ── sif: convert Docker image to Apptainer .sif ──────────────────────────────
sif: image
	@echo "Converting Docker image to Apptainer .sif at $(SIF_PATH)..."
	docker save $(IMAGE_NAME):$(IMAGE_TAG) | gzip > /tmp/autoqsar_docker.tar.gz
	apptainer build $(SIF_PATH) docker-archive:///tmp/autoqsar_docker.tar.gz
	rm -f /tmp/autoqsar_docker.tar.gz
	@# Record .sif SHA-256 in BUILD_PROVENANCE.md
	@SIF_HASH=$$(sha256sum $(SIF_PATH) | cut -d' ' -f1); \
	echo "" >> BUILD_PROVENANCE.md; \
	echo "## .sif SHA-256" >> BUILD_PROVENANCE.md; \
	echo "" >> BUILD_PROVENANCE.md; \
	echo "sif_sha256: $$SIF_HASH" >> BUILD_PROVENANCE.md; \
	echo "sif_path: $(SIF_PATH)" >> BUILD_PROVENANCE.md; \
	echo ".sif SHA-256: $$SIF_HASH"

# ── smoke: CPU smoke test inside container ────────────────────────────────────
smoke:
	@if [ ! -f "$(SIF_PATH)" ]; then \
		echo "ERROR: $(SIF_PATH) not found. Run 'make sif' first."; \
		exit 1; \
	fi
	bash tests/smoke_cpu.sh

# ── manifests: split full manifest into GPU/CPU sub-manifests ────────────────
manifests:
	@echo "Generating GPU and CPU sub-manifests from hpc/task_manifest.tsv..."
	@head -1 hpc/task_manifest.tsv | cut -f1-2 > hpc/task_manifest_gpu.tsv
	@awk -F'\t' 'NR>1 && $$3=="gpu" {print $$1"\t"$$2}' hpc/task_manifest.tsv >> hpc/task_manifest_gpu.tsv
	@head -1 hpc/task_manifest.tsv | cut -f1-2 > hpc/task_manifest_cpu.tsv
	@awk -F'\t' 'NR>1 && $$3=="cpu" {print $$1"\t"$$2}' hpc/task_manifest.tsv >> hpc/task_manifest_cpu.tsv
	@GPU_TASKS=$$(awk 'NR>1' hpc/task_manifest_gpu.tsv | wc -l); \
	CPU_TASKS=$$(awk 'NR>1' hpc/task_manifest_cpu.tsv | wc -l); \
	echo "GPU tasks: $$GPU_TASKS  (update --array=0-$$((GPU_TASKS-1)) in submit_multiseed.sbatch)"; \
	echo "CPU tasks: $$CPU_TASKS  (update --array=0-$$((CPU_TASKS-1)) in submit_multiseed_cpu.sbatch)"

# ── benchmark-a100: run or resume full A100 benchmark ────────────────────────
# Fresh run:   make benchmark-a100
# Resume run:  make benchmark-a100 BENCHMARK_OUTPUT_DIR=benchmark_results/autoqsar_benchmark_<timestamp>
benchmark-a100:
	@if [ -n "$(BENCHMARK_OUTPUT_DIR)" ]; then \
		echo "Resuming benchmark run at: $(BENCHMARK_OUTPUT_DIR)"; \
		AUTOQSAR_CONDA_ENV=$(CONDA_ENV) bash js2/run_a100_benchmark.sh \
			--output-dir "$(BENCHMARK_OUTPUT_DIR)"; \
	else \
		echo "Starting fresh A100 benchmark run..."; \
		AUTOQSAR_CONDA_ENV=$(CONDA_ENV) bash js2/run_a100_benchmark.sh; \
	fi

# ── clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf $(VENV_DIR) /tmp/autoqsar_docker.tar.gz
	@echo "Cleaned. Note: $(SIF_PATH) and Docker images not removed automatically."
