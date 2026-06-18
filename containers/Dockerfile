# AutoQSAR container — CUDA 12.1 / Ubuntu 22.04
#
# Multi-stage build:
#   builder — installs the full Python environment from pinned lockfile
#   runtime — copies the venv and source into a slim CUDA runtime image
#
# Usage (local):
#   docker build -t autoqsar:latest -f containers/Dockerfile .
#
# Convert to Apptainer .sif (see containers/autoqsar.def for direct Apptainer build):
#   docker save autoqsar:latest | gzip > autoqsar.tar.gz
#   apptainer build autoqsar.sif docker-archive://autoqsar.tar.gz

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# CUDA 12.1 devel stage is needed so that packages with CUDA extensions
# (e.g. DGL) can compile native wheels at install time if pre-built wheels
# are unavailable. The runtime stage uses the smaller cudnn8-runtime image.

ARG PYTHON_VERSION=3.11.9
ARG DEBIAN_FRONTEND=noninteractive

# System deps: Python 3.11, build tools, git for provenance
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Create a venv at /opt/venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip/setuptools/wheel once inside the venv
RUN pip install --no-cache-dir --upgrade pip==24.2 setuptools==70.2.0 wheel==0.46.3

# Install PyTorch + CUDA 12.1 wheels FIRST (separate index)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    torchdata==0.9.0

# Install DGL with CUDA 12.1 wheels
RUN pip install --no-cache-dir \
    dgl==2.2.1 \
    dgllife==0.3.2 \
    -f https://data.dgl.ai/wheels/torch-2.5/cu121/repo.html

# Install remainder of pinned deps from lockfile
# (No network version resolution — all versions are pinned)
COPY environment/requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r /tmp/requirements-lock.txt

# Install vendored PyTDC from bundled tarball
COPY .cache/pytdc-1.1.15.tar.gz /tmp/pytdc-1.1.15.tar.gz
RUN pip install --no-cache-dir /tmp/pytdc-1.1.15.tar.gz

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ARG BUILD_DATE
ARG GIT_COMMIT=unknown
ARG GIT_TAG=unknown

LABEL org.opencontainers.image.title="AutoQSAR" \
      org.opencontainers.image.description="Portable QSAR modeling and benchmarking — CUDA 12.1" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${GIT_TAG}" \
      autoqsar.cuda_version="12.1" \
      autoqsar.python_version="3.11"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        libgomp1 \
        git \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy repo source (data, benchmark_results, model_cache are bind-mounted at runtime)
COPY --chown=nobody:nogroup . /opt/autoqsar/

WORKDIR /opt/autoqsar

# Reproducibility env
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    OMP_NUM_THREADS=1

ENTRYPOINT ["python", "/opt/autoqsar/run_one.py"]
CMD ["--help"]
