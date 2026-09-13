###############################################################################
# FlashHead — Streaming REST API  (server2.py)                                #
# ─────────────────────────────────────────────────────────────────────────── #
#  CUDA 12.8 · cuDNN 9 · Python 3.10 · torch 2.7.1+cu128 · flash-attn 2.8.0    #
#                                                                             #
#  Model weights are NOT included. Mount host checkpoints at /checkpoints and #
#  select them at runtime through POST /model/load.                            #
#                                                                             #
#  Build (from repo root):                                                     #
#      docker build -f docker/Dockerfile -t flashhead-api:latest .            #
#  Run:                                                                        #
#      docker run --rm --gpus all -p 8000:8000 --shm-size=8g \                #
#          -v "$(pwd)/models:/checkpoints:ro" flashhead-api                   #
###############################################################################
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# ── system dependencies ──────────────────────────────────────────────────────
#   Ubuntu 22.04 ships Python 3.10 (matches the cp310 flash-attn wheel).
#   ffmpeg  → segment muxing (video + audio)
#   libGL / glib / gomp → shared libs opencv, mediapipe & decord dlopen on import
# (drop the NVIDIA CUDA apt repo — it can 403, and we install CUDA via pip wheels,
#  not apt, so only the standard Ubuntu archives are needed here)
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ffmpeg git curl ca-certificates \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1) torch stack from the CUDA-12.8 wheel index ────────────────────────────
#     (+cu128 local version outranks the plain PyPI 2.7.1, so the CUDA build is
#      chosen; the assert guards against silently getting the CPU wheel.)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
        --timeout 300 \
        --retries 10 \
        --resume-retries 10 \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1+cu128 \
        torchvision==0.22.1+cu128 \
 && python -c "import torch, torchvision; print('torch', torch.__version__); assert torch.version.cuda == '12.8', torch.version.cuda"


# ── 2) xformers 0.0.31 — PyPI abi3 wheel, compiled for CUDA 12.8 ──────────────
#     (--no-deps: torch/numpy already pinned; nothing else to resolve)
RUN python -m pip install --no-deps xformers==0.0.31   -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 3) flash-attention — prebuilt wheel (cu12 · torch2.7 · cxx11abiTRUE · cp310)
#RUN python -m pip install --no-deps \
#        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# ── 3) flash-attention — prebuilt wheel (cu12 · torch2.7 · cxx11abiTRUE · cp310)
# ── 3) flash-attention — prebuilt wheel (cu12 · torch2.7 · cxx11abiTRUE · cp310)
RUN apt-get update && apt-get install -y --no-install-recommends aria2 \
    && aria2c -x 16 -s 16 -k 1M -d /tmp \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" \
    && python -m pip install --no-deps /tmp/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
    && rm -f /tmp/flash_attn-*.whl \
    && apt-get purge -y aria2 && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ── 4) externally managed model weights ──────────────────────────────────────
ENV CKPT_DIR=/checkpoints/SoulX-FlashHead-1_3B \
    WAV2VEC_DIR=/checkpoints/wav2vec2-base-960h \
    AUTOLOAD=0

# Model checkpoints are managed outside the image. Mount a host directory at
# /checkpoints and load its checkpoint paths through POST /model/load.
RUN mkdir -p /checkpoints

# ── 5) remaining pinned dependencies (known-good freeze) ─────────────────────
COPY docker/requirements-lock.txt /tmp/requirements-lock.txt
RUN python -m pip install -r /tmp/requirements-lock.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# ── 6) C/C++ toolchain — torch.compile (inductor) shells out to a compiler at
#     runtime. The CUDA *runtime* base image ships without one, so the first
#     /session call dies with "Failed to find C compiler".
#     Deliberately placed BELOW the weights + deps: adding it here costs one
#     small layer instead of re-triggering the ~16 GB of downloads above.
#     Fold it into the apt block at the top on the next from-scratch build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── application code ─────────────────────────────────────────────────────────
COPY flash_head/ ./flash_head/
COPY server2.py test_stream.py generate_video.py ./
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

# Liveness: /model/status answers only once uvicorn is up (after model load).
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=5 \
    CMD curl -fsS http://localhost:8000/model/status || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
