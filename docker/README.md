# FlashHead API — Docker (Linux / WSL2 + GPU)

Containerized build of the **FlashHead streaming REST API** (`server2.py`).
See [`../README_API.md`](../README_API.md) for the API itself.

The image bundles the full CUDA/PyTorch runtime; the **model weights are mounted at
runtime** (≈15 GB — never baked into the image).

| | |
|---|---|
| Base | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` |
| Python | 3.10 |
| Torch | `2.7.1+cu128` · torchvision `0.22.1` · xformers `0.0.31` |
| Attention | `flash-attn 2.8.0.post2` (prebuilt `cu12torch2.7cxx11abiTRUE-cp310` wheel) |
| Deps | pinned from a known-good environment → [`requirements-lock.txt`](./requirements-lock.txt) |
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

The `models/` directory is excluded via [`.dockerignore`](../.dockerignore), so the build
context stays a few MB. First build downloads the CUDA base + torch + flash-attn (several
GB); subsequent builds are cached.

---

## 3. Get the model weights

The weights are **not** in the image — download them once on the host into `models/`,
then mount that directory into the container (next step).

| Model | Source |
|-------|--------|
| `SoulX-FlashHead-1_3B` | 🤗 [huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| `wav2vec2-base-960h` | 🤗 [huggingface.co/facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) |

```bash
pip install "huggingface_hub[cli]"
# run from the repo root so the paths match the mount below
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h     --local-dir ./models/wav2vec2-base-960h
```

You end up with `models/SoulX-FlashHead-1_3B/` (contains `Model_Lite/` and `Model_Pro/`)
and `models/wav2vec2-base-960h/` — the two directories the entrypoint expects.

---

## 4. Run

The container needs the GPU and the model weights mounted at `/app/models`:

```bash
docker run --rm --gpus all \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    --shm-size=8g \
    --name flashhead-api \
    flashhead-api:latest
```

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

## 5. Configuration (environment variables)

Pass with `-e VAR=value` (or the `environment:` block in Compose):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TYPE` | `lite` | `lite` (fast) or `pro` (higher quality) |
| `PORT` | `8000` | Server port (also update `-p`) |
| `CKPT_DIR` | `models/SoulX-FlashHead-1_3B` | Checkpoint dir (inside the container) |
| `WAV2VEC_DIR` | `models/wav2vec2-base-960h` | Wav2Vec2 dir |
| `IDLE_TIMEOUT` | `300` | Seconds before an idle session is auto-deleted |
| `GC_INTERVAL` | `60` | Session GC check interval |
| `AUTOLOAD` | `1` | `0` = start without loading; load later via `POST /model/load` |

```bash
# example: run the PRO model
docker run --rm --gpus all -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" --shm-size=8g \
    -e MODEL_TYPE=pro \
    flashhead-api:latest
```

---

## 6. Test the container

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

## 7. Notes & troubleshooting

- **`could not select device driver "" with capabilities: [[gpu]]`** → the NVIDIA Container
  Toolkit isn't installed/configured; redo step 1 and restart Docker.
- **First request is slow (~80 s)** — one-time CUDA warmup; every later request is ~2 s to
  first segment. Send a throwaway request after startup to pre-warm.
- **`AUTOLOAD=1 but model directories were not found`** — you didn't mount the weights.
  Add `-v "$(pwd)/models:/app/models:ro"`, or set `-e AUTOLOAD=0` to load via the API.
- **One model / one GPU** — inference is serialized on a single pipeline lock. For
  concurrency run multiple containers (one GPU each) behind a load balancer.
- **Image size** (~15 GB) is mostly the CUDA + torch runtime. `/dev/shm` is raised to 8 GB
  (`--shm-size=8g`) to avoid PyTorch shared-memory errors.
- **Driver/CUDA:** the container's CUDA 12.8 userspace runs on the host driver via the
  toolkit — the same combination already working on this machine bare-metal.
