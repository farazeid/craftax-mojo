# Benchmark image for the FLAIR NVIDIA nodes: Mojo/MAX + JAX (CUDA 12).
# Build and run as your own user (FLAIR rule 1); mount the repo at /app.
#
#   docker build -f docker/Dockerfile --build-arg UID=$(id -u) \
#       --build-arg GID=$(id -g) -t ${USER}_craftax-mojo .
#   docker run --rm --gpus '"device=0"' -v $(pwd):/app ${USER}_craftax-mojo \
#       bash scripts/bench_all.sh
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

ARG UID
ARG GID
RUN groupadd -o -g $GID myuser && useradd -u $UID -g $GID -m myuser
USER myuser
WORKDIR /home/myuser/env

# Dependencies only; the repo itself is mounted at /app at run time.
COPY --chown=myuser:myuser pyproject.toml uv.lock ./
ENV UV_PROJECT_ENVIRONMENT=/home/myuser/venv
RUN uv sync --frozen --extra cuda --no-install-project
ENV PATH=/home/myuser/venv/bin:$PATH
# Mojo's bundled PTX toolchain can be newer than a node's driver supports
# (e.g. CUDA 12.8 on flair-node-13); use ptxas 12.9 from the CUDA 12 wheels.
ENV MODULAR_NVPTX_COMPILER_PATH=/home/myuser/venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas
WORKDIR /app
