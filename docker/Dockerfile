###############################################################################
# FlashHead — Streaming REST API  (server2.py)                                #
# ─────────────────────────────────────────────────────────────────────────── #
#  CUDA 12.8 · cuDNN 9 · Python 3.10 · torch 2.7.1+cu128 · flash-attn 2.8.0    #
#                                                                             #
#  Model weights are DOWNLOADED AT BUILD TIME into /app/models — no manual     #
#  download, no host mount needed. Select what gets baked in with             #
#  --build-arg MODEL_VARIANT=lite|pro|all   (default: lite, ≈8 GB of weights) #
#                                                                             #
#  Build (from repo root):                                                     #
#      docker build -f docker/Dockerfile -t flashhead-api:latest .            #
#      docker build -f docker/Dockerfile --build-arg MODEL_VARIANT=all \      #
#          -t flashhead-api:all .                                             #
#  Run:                                                                        #
#      docker run --rm --gpus all -p 8000:8000 --shm-size=8g flashhead-api    #
#                                                                             #
#  Prefer the old behaviour (weights mounted from the host)? Build with       #
#      --build-arg DOWNLOAD_MODELS=0                                          #
#  and run with  -v "$(pwd)/models:/app/models:ro".                           #
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
RUN python -m pip install \
        --timeout 300 --retries 20 --resume-retries 20 \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        torch==2.7.1 torchvision==0.22.1 \
 && python -c "import torch, torchvision; print('torch', torch.__version__); assert torch.version.cuda == '12.8', torch.version.cuda"

# ── 2) xformers 0.0.31 — PyPI abi3 wheel, compiled for CUDA 12.8 ──────────────
#     (--no-deps: torch/numpy already pinned; nothing else to resolve)
RUN python -m pip install --no-deps xformers==0.0.31

# ── 3) flash-attention — prebuilt wheel (cu12 · torch2.7 · cxx11abiTRUE · cp310)
RUN python -m pip install --no-deps \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

# ── 4) model weights — downloaded here, straight into the image ──────────────
#     Kept ABOVE the app layers so editing code (or the dep lock) never forces
#     a re-download. Only what the chosen variant actually loads is fetched:
#       lite → Model_Lite + VAE_LTX   pro → Model_Pro + VAE_Wan   all → both
#     (`assets/` are demo videos; wav2vec2 ships torch + TF + bin copies of the
#      same weights — transformers loads model.safetensors, so skip the rest.)
ARG DOWNLOAD_MODELS=1
ARG MODEL_VARIANT=lite
ARG HF_FLASHHEAD_REPO=Soul-AILab/SoulX-FlashHead-1_3B
ARG HF_WAV2VEC_REPO=facebook/wav2vec2-base-960h

# hf_transfer = rust-backed multipart downloader; ~15 GB over a single stream
# is otherwise the slowest part of the build by far.
RUN python -m pip install "huggingface_hub==0.36.2" hf_transfer

ENV CKPT_DIR=/app/models/SoulX-FlashHead-1_3B \
    WAV2VEC_DIR=/app/models/wav2vec2-base-960h

# -f (noglob): $inc must word-split into separate patterns, but the shell must
# not try to expand `Model_Lite/*` against the local filesystem.
RUN set -euf; \
    if [ "$DOWNLOAD_MODELS" != "1" ]; then \
        echo "[build] DOWNLOAD_MODELS=0 — skipping, mount weights at /app/models"; \
        mkdir -p /app/models; exit 0; \
    fi; \
    case "$MODEL_VARIANT" in \
      lite) inc='Model_Lite/* VAE_LTX/*' ;; \
      pro)  inc='Model_Pro/* VAE_Wan/*'  ;; \
      all)  inc='Model_Lite/* VAE_LTX/* Model_Pro/* VAE_Wan/*' ;; \
      *)    echo "[build] ERROR: MODEL_VARIANT must be lite|pro|all (got '$MODEL_VARIANT')" >&2; exit 1 ;; \
    esac; \
    export HF_HUB_ENABLE_HF_TRANSFER=1; \
    echo "[build] downloading $HF_FLASHHEAD_REPO ($MODEL_VARIANT)"; \
    hf download "$HF_FLASHHEAD_REPO" \
        --local-dir "$CKPT_DIR" \
        --include $inc '*.json' ; \
    echo "[build] downloading $HF_WAV2VEC_REPO"; \
    hf download "$HF_WAV2VEC_REPO" \
        --local-dir "$WAV2VEC_DIR" \
        --exclude 'tf_model.h5' 'pytorch_model.bin' ; \
    rm -rf "$CKPT_DIR/.cache" "$WAV2VEC_DIR/.cache"; \
    du -sh "$CKPT_DIR" "$WAV2VEC_DIR"

# ── 5) remaining pinned dependencies (known-good freeze) ─────────────────────
COPY docker/requirements-lock.txt /tmp/requirements-lock.txt
RUN python -m pip install -r /tmp/requirements-lock.txt

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
