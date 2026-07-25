# FlashHead API — Docker (Linux / WSL2 + GPU)

Containerized build of the **FlashHead streaming REST API** (`server2.py`).
See [`../README_API.md`](../README_API.md) for the API itself.

The image bundles the full CUDA/PyTorch runtime **and the model weights** — the Dockerfile
downloads them from Hugging Face during `docker build`, so there is nothing to fetch by hand
and nothing to mount at runtime.

| | |
|---|---|
| Base | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` |
| Python | 3.10 |
| Torch | `2.7.1+cu128` · torchvision `0.22.1` · xformers `0.0.31` |
| Attention | `flash-attn 2.8.0.post2` (prebuilt `cu12torch2.7cxx11abiTRUE-cp310` wheel) |
| Deps | pinned from a known-good environment → [`requirements-lock.txt`](./requirements-lock.txt) |
| Weights | downloaded at build time into `/app/models` (`MODEL_VARIANT=lite\|pro\|all`) |
| Exposes | `8000` |

---

## 1. Prerequisites (WSL2)

You already run FlashHead bare-metal in WSL, so the GPU is set up. For Docker you also need
the **NVIDIA Container Toolkit** inside the WSL distro so containers can see the GPU:

```bash
# one-time: install the NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker      # or: Docker Desktop → restart

# verify the container can see the GPU
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

> Docker Desktop users: enable the WSL2 backend and GPU support. The Windows NVIDIA driver
> provides the GPU to WSL — do **not** install a driver inside the container.

---

## 2. Build

From the **repo root** (the build context must include `flash_head/` and `server2.py`):

```bash
docker build -f docker/Dockerfile -t flashhead-api:latest .
```

The local `models/` directory is excluded via [`.dockerignore`](../.dockerignore), so the
build context stays a few MB — the weights come from Hugging Face inside the build, not from
your disk. The first build pulls the CUDA base + torch + flash-attn and the weights (tens of
GB, expect a while); subsequent builds are cached, and editing code does **not** re-download
the weights.

### Which weights get baked in

| Model | Source |
|-------|--------|
| `SoulX-FlashHead-1_3B` | 🤗 [huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| `wav2vec2-base-960h` | 🤗 [huggingface.co/facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) |

Only what the selected variant actually loads is fetched:

| `--build-arg MODEL_VARIANT=` | Fetched from the checkpoint repo | Weights size |
|---|---|---|
| `lite` *(default)* | `Model_Lite/` + `VAE_LTX/` | ≈ 7.3 GB |
| `pro` | `Model_Pro/` + `VAE_Wan/` | ≈ 6.2 GB |
| `all` | both of the above | ≈ 13.5 GB |

plus ≈ 0.4 GB of wav2vec2 (`model.safetensors` only — the TF/`.bin` duplicates are skipped).

```bash
# image that can serve both lite and pro
docker build -f docker/Dockerfile --build-arg MODEL_VARIANT=all -t flashhead-api:all .
```

`MODEL_VARIANT` must cover whatever `MODEL_TYPE` you run with — the entrypoint checks this on
boot and tells you to rebuild rather than failing halfway through model load.

### Building without the weights

To keep the old behaviour (slim image, weights mounted from the host):

```bash
docker build -f docker/Dockerfile --build-arg DOWNLOAD_MODELS=0 -t flashhead-api:slim .

# then fetch the weights on the host, from the repo root
pip install huggingface_hub
hf download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
hf download facebook/wav2vec2-base-960h     --local-dir ./models/wav2vec2-base-960h
```

then add `-v "$(pwd)/models:/app/models:ro"` to the `docker run` below.

---

## 3. Run

The container needs the GPU; the weights are already inside it:

```bash
docker run --rm --gpus all \
    -p 8000:8000 \
    --shm-size=8g \
    --name flashhead-api \
    flashhead-api:latest
```

> Do **not** mount `-v .../models:/app/models` on a normal build — an empty or partial host
> directory shadows the baked-in weights and the container will refuse to start.

On boot it auto-loads the **lite** model and serves on `:8000`. When you see the model
finish loading, it's ready. Check it:

```bash
curl http://localhost:8000/model/status
# {"loaded":true,"model_type":"lite", ...}
```

Open the interactive docs at **http://localhost:8000/docs**.

### Or with Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## 4. Configuration (environment variables)

Pass with `-e VAR=value` (or the `environment:` block in Compose):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TYPE` | `lite` | `lite` (fast) or `pro` (higher quality) — must be covered by the image's `MODEL_VARIANT` |
| `PORT` | `8000` | Server port (also update `-p`) |
| `CKPT_DIR` | `/app/models/SoulX-FlashHead-1_3B` | Checkpoint dir (inside the container) |
| `WAV2VEC_DIR` | `/app/models/wav2vec2-base-960h` | Wav2Vec2 dir |
| `IDLE_TIMEOUT` | `300` | Seconds before an idle session is auto-deleted |
| `GC_INTERVAL` | `60` | Session GC check interval |
| `AUTOLOAD` | `1` | `0` = start without loading; load later via `POST /model/load` |

```bash
# example: run the PRO model (image must be built with MODEL_VARIANT=pro or =all)
docker run --rm --gpus all -p 8000:8000 --shm-size=8g \
    -e MODEL_TYPE=pro \
    flashhead-api:all
```

---

## 5. Test the container

The API is on `localhost:8000`, so drive it **from the host** exactly like the bare-metal
server (`test_stream.py` targets port 8000):

```bash
# first frame of the sample video → face image, then stream the audio
ffmpeg -y -i examples/example.mp4 -vframes 1 examples/first_frame.png
python test_stream.py live examples/first_frame.png examples/example.wav output.mp4
```

Or with cURL:

```bash
BASE=http://localhost:8000
SESSION=$(curl -s -X POST $BASE/session \
    -F "image=@examples/first_frame.png" -F "use_face_crop=true" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
curl -X POST "$BASE/session/$SESSION/generate" \
    -F "audio=@examples/example.wav" --output stream_output.bin
python test_stream.py split stream_output.bin output.mp4
```

**Verified result** (RTX 4090, lite model, in-container): the sample `example.wav` (36.5 s)
streams back as 13 MP4 segments; warm generation ≈ **11 s (~3.2× real-time)**. See
[`../README_API.md`](../README_API.md) for the full benchmark.

---

## 6. Notes & troubleshooting

- **`could not select device driver "" with capabilities: [[gpu]]`** → the NVIDIA Container
  Toolkit isn't installed/configured; redo step 1 and restart Docker.
- **First request is slow (~80 s)** — one-time CUDA warmup; every later request is ~2 s to
  first segment. Send a throwaway request after startup to pre-warm.
- **`AUTOLOAD=1 but model directories were not found`** — on a normal build the weights are
  in the image, so this means a host `-v .../models:/app/models` mount is shadowing them
  (drop it), or the image was built with `DOWNLOAD_MODELS=0` (then the mount is required).
  `-e AUTOLOAD=0` boots without a model and lets you load one via the API.
- **`MODEL_TYPE=pro needs .../Model_Pro`** — the image was built `lite`-only. Rebuild with
  `--build-arg MODEL_VARIANT=pro` (or `=all`).
- **Build fails downloading weights** — Hugging Face rate limit or a dropped connection.
  Re-run the build (finished layers are cached); the download step resumes from scratch but
  the CUDA/torch layers do not rebuild.
- **One model / one GPU** — inference is serialized on a single pipeline lock. For
  concurrency run multiple containers (one GPU each) behind a load balancer.
- **Image size** (~23 GB for `lite`, ~29 GB for `all`) — CUDA + torch runtime plus the baked
  weights. Build with `--build-arg DOWNLOAD_MODELS=0` for the ~15 GB weightless image.
  `/dev/shm` is raised to 8 GB (`--shm-size=8g`) to avoid PyTorch shared-memory errors.
- **Driver/CUDA:** the container's CUDA 12.8 userspace runs on the host driver via the
  toolkit — the same combination already working on this machine bare-metal.
