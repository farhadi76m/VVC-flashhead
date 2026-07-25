# FlashHead API — Docker

Talking-head video generation. Send a face image + audio, get MP4 video segments
streamed back as they are generated.

- **Model:** lite — baked into the image, nothing to download at runtime
- **Face crop:** on by default, nothing to configure
- **Port:** 8000

---

## 1. Requirements

- NVIDIA GPU with ~12 GB free VRAM
- NVIDIA driver 525 or newer
- Docker with GPU access

Check the GPU is visible:

```bash
nvidia-smi
```

**Pick your compose file.** Every command below uses `docker/docker-compose.yml`,
which is set up for **Docker Desktop on WSL2**. On a **native Linux** host use
`docker/docker-compose-linux.yml` instead — same settings, different GPU
passthrough. See [the note at the end](#note-on-gpu-passthrough).

---

## 2. Build

Run from the repo root, **not** from `docker/`:

```bash
docker compose -f docker/docker-compose.yml build
```

Takes 20–40 minutes and downloads ~7.7 GB of weights into the image. Final image
is about 23 GB. You only do this once.

---

## 3. Run

```bash
docker compose -f docker/docker-compose.yml up -d
```

The model loads at startup and is ready in about **20 seconds**.

**Important:** the *first* generate request is slow — around **2 minutes** —
because PyTorch compiles the model on first use. Every request after that is
fast (~15 seconds). Send one throwaway request after startup so the first real
user does not wait.

---

## 4. Check it works

```bash
curl http://localhost:8000/model/status
```

Expected:

```json
{"loaded":true,"model_type":"lite", ...}
```

`"loaded":true` means it is ready. Interactive API docs: <http://localhost:8000/docs>

---

## 5. Use the API

Three steps: create a session with a face image, send audio, delete the session.

### Create a session

```bash
curl -X POST http://localhost:8000/session \
  -F "image=@face.png"
```

```json
{"session_id":"a1b2c3d4-...","status":"ready"}
```

Keep the `session_id`. One session = one person's face. Reuse it for every turn
of the conversation — the image is only processed once.

### Generate video from audio

```bash
curl -X POST http://localhost:8000/session/<SESSION_ID>/generate \
  -F "audio=@speech.wav" \
  -o output.bin
```

Audio should be **WAV, 16 kHz, mono**.

The response is a `multipart/x-mixed-replace` stream. Each part is a complete MP4
covering ~3 seconds of video, sent as soon as it is ready — so playback can start
before the full clip is finished. Each part carries an `X-Segment-Index` header.

Measured on an RTX 4090: ~40 seconds of speech → 13 segments in **14 seconds**
(after warmup). Generation is faster than real time, so playback keeps up.

### Delete the session

```bash
curl -X DELETE http://localhost:8000/session/<SESSION_ID>
```

Idle sessions are cleaned up automatically after 5 minutes.

### Working example

`test_stream.py` shows how to consume the stream and joins the segments into one
playable file:

```bash
python test_stream.py live face.png speech.wav
```

---

## 6. Day-to-day commands

```bash
# logs
docker compose -f docker/docker-compose.yml logs -f

# stop
docker compose -f docker/docker-compose.yml down

# start
docker compose -f docker/docker-compose.yml up -d

# restart
docker compose -f docker/docker-compose.yml restart
```

After changing code (`server2.py`, `flash_head/`), rebuild and restart:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
```

Code changes rebuild in under a minute — the weights are cached and are not
downloaded again.

---

## 7. All endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/model/status` | Is the model loaded |
| `POST` | `/session` | Create session (upload face image) |
| `POST` | `/session/{id}/generate` | Audio in → video segments out |
| `PATCH` | `/session/{id}/image` | Change the face mid-session |
| `GET` | `/session/{id}/status` | Session info |
| `DELETE` | `/session/{id}` | Delete session |
| `GET` | `/sessions` | List active sessions |

---

## 8. Troubleshooting

**`{"loaded":false}` or connection refused**

Still starting — give it ~20 seconds. If it stays down, check the logs:

```bash
docker compose -f docker/docker-compose.yml logs -f
```

**First request takes ~2 minutes**

Normal. PyTorch compiles the model on first use; later requests take ~15 s.
See section 3.

**Port 8000 already in use**

Edit `docker/docker-compose.yml` and change the left-hand number:

```yaml
ports:
  - "8100:8000"
```

**GPU not detected / CUDA errors**

Confirm the container sees the GPU:

```bash
docker exec flashhead-api python -c "import torch; print(torch.cuda.is_available())"
```

Must print `True`. If it prints `False`, see the note below.

**Out of memory**

Run only one generate request at a time per GPU. Lite needs ~12 GB.

---

## Note on GPU passthrough

There are two compose files — pick the one that matches your host:

| Host | File |
|---|---|
| Docker Desktop on **WSL2** | `docker/docker-compose.yml` |
| **Native Linux** server | `docker/docker-compose-linux.yml` |

They differ only in how the GPU is handed to the container.

WSL2 exposes the GPU as `/dev/dxg` instead of `/dev/nvidia*`, so the WSL2 file
passes that device directly rather than using `gpus: all` — the NVIDIA container
hook misdetects the mode on WSL and the container ends up with no usable GPU. It
also mounts `/usr/lib/wsl` and sets `LD_LIBRARY_PATH` so the WSL driver libs win
over the image's CUDA compat libs.

The Linux file uses `gpus: all` and none of those workarounds — on a native host
they would break CUDA instead of fixing it. It needs the NVIDIA Container Toolkit
installed:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# verify
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

`gpus:` needs Docker Compose v2.30 or newer; the file carries a commented
`deploy:` block to use instead on older versions.

The comments in both files explain the same thing.
