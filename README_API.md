# FlashHead — Streaming REST API (`server2.py`)

Real-time **audio → talking-head video** service for **SoulX-FlashHead**, built with
FastAPI. You upload a face image once, then stream audio in; the server streams back
**MP4 video segments as each one is generated** (no waiting for the whole clip).

> This document covers **`server2.py`** — the streaming, session-based API.
> For the model itself see [`README.md`](./README.md).

---

## ✨ What it does

```
POST /model/load            → load FlashHead + Wav2Vec2 into the GPU (once)
POST /session               → create a session + upload a face image
PATCH /session/{id}/image   → hot-swap the face at any time
POST /session/{id}/generate → send audio, receive an MP4 stream (1 part per ~3 s segment)
DELETE /session/{id}        → free resources
```

- **Streaming:** the response is a `multipart/x-mixed-replace` stream — each ~3-second
  MP4 segment is pushed the instant it is rendered, so playback can start almost
  immediately.
- **Stateful sessions:** the audio context window (deque) carries across turns, so a
  multi-turn conversation stays temporally coherent. Idle sessions are garbage-collected
  after 5 minutes.
- **Portrait preserved:** with `use_face_crop=true` the face is detected, animated, and
  wrapped back into the **original full-resolution frame**, so the output keeps the input
  aspect ratio (e.g. 720×1280).

---

## ✅ Tested example & results

The API was tested end-to-end on the bundled example, using the **first frame of
`examples/example.mp4`** as the face image (extracted with ffmpeg, as requested) and
**`examples/example.wav`** as the driving audio.

### Inputs

| Input | Source | Details |
|-------|--------|---------|
| Face image | first frame of `examples/example.mp4` | 720 × 1280, extracted via ffmpeg |
| Audio | `examples/example.wav` | 36.25 s, 48 kHz stereo (resampled to 16 kHz mono internally) |
| Model | `lite` | `models/SoulX-FlashHead-1_3B` + `models/wav2vec2-base-960h` |
| GPU | NVIDIA RTX 4090 (24 GB) | single GPU |

### Commands that produced the result

```bash
# 1. Extract the first frame of the video to use as the face image
ffmpeg -y -i examples/example.mp4 -vframes 1 examples/first_frame.png

# 2. Server already running on :8000 (see "Start the server" below)

# 3. Drive the full streaming flow and reassemble the segments
python test_stream.py live examples/first_frame.png examples/example.wav \
    sample_results/example_output.mp4
```

### Output

- 📄 **`sample_results/example_output.mp4`** — 720 × 1280, **25 fps**, H.264 video +
  16 kHz AAC audio, **36.5 s**, 912 frames, ~2.0 MB, delivered as **13 streamed segments**.

![Generated frames](sample_results/example_filmstrip.png)

*Frames sampled across the generated clip — the face is lip-synced to the audio with
natural blinks and head motion, while the suit and background from the source frame are
preserved.*

### Performance

The **first** request after the model loads pays a one-time CUDA/kernel warmup cost.
Every request after that runs at steady state:

| Metric | Cold start (1st request) | Warm (steady state) |
|--------|--------------------------|---------------------|
| Session create (image + face crop) | 12.3 s | ~0.0 s |
| Time to **first** segment | ~81 s *(incl. warmup)* | **1.9 s** |
| Total generation time | ~90 s | **11.3 s** |
| Video produced | 36.5 s | 36.5 s |
| Real-time factor | 0.4× | **~3.2× faster than real-time** |
| End-to-end throughput | — | **~80 FPS** (render + encode + stream) |

In warm mode a new ~3-second segment arrives roughly **every 0.8 s**, so the client keeps
a comfortable playback buffer while generation runs ahead of playback.

> 💡 To avoid the cold-start penalty in production, send one short "warmup" request right
> after `POST /model/load`.

---

## 🔧 Prerequisites

1. **Environment** — the `flash_head` package and its dependencies installed
   (`pip install -r requirements.txt`), plus a CUDA GPU and `ffmpeg`/`ffprobe` on `PATH`.
2. **Model weights** present locally:
   - `models/SoulX-FlashHead-1_3B/`
   - `models/wav2vec2-base-960h/`
3. **Python API deps:** `fastapi`, `uvicorn`, `python-multipart`, `librosa`, `imageio`,
   `torch`, `loguru` (mostly already pulled in by `requirements.txt`).

---

## 🚀 Start the server

Load the model automatically on boot and listen on port **8000**:

```bash
python server2.py \
    --ckpt-dir     models/SoulX-FlashHead-1_3B/ \
    --wav2vec-dir  models/wav2vec2-base-960h/ \
    --model-type   lite \
    --host 0.0.0.0 --port 8000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port |
| `--ckpt-dir` | – | FlashHead checkpoint dir (auto-loads model on start if set) |
| `--wav2vec-dir` | – | Wav2Vec2 dir |
| `--model-type` | `lite` | `lite` (fast) or `pro` (higher quality) |
| `--idle-timeout` | `300` | Seconds before an idle session is auto-deleted |
| `--gc-interval` | `60` | How often the session GC runs |

Interactive Swagger docs: **http://localhost:8000/docs**

Check status:

```bash
curl http://localhost:8000/model/status
# {"loaded":true,"model_type":"lite","ckpt_dir":"models/SoulX-FlashHead-1_3B/", ...}
```

---

## 🧑‍💻 How to use it

### Option A — quick end-to-end test (recommended)

`test_stream.py` opens a session, streams the audio, and reassembles the segments into a
single MP4, printing timing as parts arrive:

```bash
# python test_stream.py live <face_image> <audio.wav> [output.mp4]
python test_stream.py live examples/first_frame.png examples/example.wav output.mp4
```

```
[  0.1s]  session  5b42a9de-…
[  0.1s]  stream open — waiting for segments …
[  1.9s]  ▶ segment   0  ( 186,994 bytes)  →  appended  →  output.mp4  [1 seg  2.9s]
[  2.6s]  ▶ segment   1  ( 170,462 bytes)  →  appended  →  output.mp4  [2 segs 5.8s]
   …
[ 11.3s]  ✅ done  — 13 segments, 36.5 s, 2.04 MB
```

### Option B — cURL

```bash
BASE=http://localhost:8000

# 1. Create a session with the face image
SESSION=$(curl -s -X POST $BASE/session \
    -F "image=@examples/first_frame.png" \
    -F "use_face_crop=true" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "session = $SESSION"

# 2. Generate — segments stream into the file as they are produced
curl -X POST "$BASE/session/$SESSION/generate" \
    -F "audio=@examples/example.wav" \
    --output stream_output.bin

# 3. (optional) hot-swap the face for the next turn
curl -X PATCH "$BASE/session/$SESSION/image" -F "image=@examples/girl.png"

# 4. Delete the session
curl -X DELETE "$BASE/session/$SESSION"
```

The `stream_output.bin` is the raw multipart body. Split it into individual MP4s with:

```bash
python test_stream.py split stream_output.bin output.mp4
```

### Option C — minimal Python client

```python
import re, requests

BASE = "http://localhost:8000"

# 1. session
with open("examples/first_frame.png", "rb") as f:
    sid = requests.post(f"{BASE}/session", files={"image": f}).json()["session_id"]

# 2. stream audio -> MP4 segments
def iter_segments(resp):
    buf, in_body, n = b"", False, 0
    for raw in resp.iter_content(8192):
        buf += raw
        while True:
            if not in_body:
                end = buf.find(b"\r\n\r\n")
                if end == -1: break
                head, buf = buf[:end].decode("latin1"), buf[end+4:]
                n = int(re.search(r"Content-Length:\s*(\d+)", head, re.I).group(1))
                in_body = True
            else:
                if len(buf) < n: break
                yield buf[:n]; buf, in_body = buf[n:], False
                if buf.startswith(b"\r\n"): buf = buf[2:]

with open("examples/example.wav", "rb") as f:
    resp = requests.post(f"{BASE}/session/{sid}/generate",
                         files={"audio": f}, stream=True)   # stream=True is essential
    for i, mp4 in enumerate(iter_segments(resp)):
        open(f"seg_{i:04d}.mp4", "wb").write(mp4)            # play/append as it arrives
        print("segment", i, len(mp4), "bytes")

requests.delete(f"{BASE}/session/{sid}")
```

---

## 📡 API reference

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/model/load` | `ckpt_dir`, `wav2vec_dir`, `model_type` (form) | `{status, model_type}` |
| `GET`  | `/model/status` | – | model load state |
| `POST` | `/session` | `image` (file), `base_seed=9999`, `use_face_crop=true` | `{session_id, status}` |
| `PATCH`| `/session/{id}/image` | `image` (file), `base_seed`, `use_face_crop` | `{status}` |
| `POST` | `/session/{id}/generate` | `audio` (file), `use_face_crop` | **multipart MP4 stream** |
| `GET`  | `/session/{id}/status` | – | turn count, idle time |
| `DELETE`| `/session/{id}` | – | `{status: deleted}` |
| `GET`  | `/sessions` | – | list of active sessions |

### Streaming protocol

`POST /session/{id}/generate` responds with
`Content-Type: multipart/x-mixed-replace; boundary=flashhead`. Each part is a complete,
independently-playable MP4:

```
--flashhead
Content-Type: video/mp4
X-Segment-Index: 0
Content-Length: <n>

<raw mp4 bytes>
--flashhead
Content-Type: video/mp4
X-Segment-Index: 1
Content-Length: <n>

<raw mp4 bytes>
--flashhead--
```

Parse `Content-Length` to know exactly how many bytes to read for each segment, and
`X-Segment-Index` for ordering. **Read the response as a stream** (e.g. `stream=True` in
`requests`, `--output` in curl) — buffering the whole body defeats the point.

---

## ⚙️ How it works (internals)

- Audio is resampled to **16 kHz mono** and split into fixed-length inference chunks.
- **`CHUNKS_PER_SEGMENT = 3`** chunks (~3 s of video) are encoded into one MP4 segment,
  which is why parts arrive roughly every 3 seconds of content.
- A background inference thread feeds a queue; the HTTP layer drains it and muxes each
  segment's frames with its matching audio via `ffmpeg` (H.264 video + AAC audio).
- Model inference is guarded by a global lock (one GPU, one pipeline), so requests are
  serialized on the model — run multiple workers/GPUs for concurrency.
- A daemon GC thread deletes sessions idle longer than `--idle-timeout`.

Key inference parameters (from `get_infer_params()`): `frame_num=33`, `tgt_fps=25`,
`sample_rate=16000`, `cached_audio_duration=8 s`, model canvas `512×512`.

---

## 🩹 Notes & gotchas

- **First request is slow** (~80 s here) due to one-time CUDA warmup — subsequent requests
  are fast (~1.9 s to first segment). Warm the model with a throwaway request at startup.
- `server2.py` serializes inference on a single pipeline lock; for concurrent users scale
  horizontally (more processes/GPUs) behind a load balancer.
- The bundled `client.py` points at port **8002** and contains a leftover `breakpoint()` —
  prefer `test_stream.py` or the snippet above, which target port **8000**.
- Give `POST /session/{id}/generate` a **`.wav`** file; other formats are decoded by
  `librosa` but 16 kHz mono WAV is the tested path.
