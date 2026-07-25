"""
FlashHead · Streaming REST API
══════════════════════════════════════════════════════════════════════════

Flow
────
  1.  POST /model/load          →  load model into GPU (once)
  2.  POST /session             →  create session + upload face image
  3.  PATCH /session/{id}/image →  swap image any time
  4.  POST /session/{id}/generate   ← send audio
                                    → multipart stream, one MP4 part per segment
                                      parts arrive AS EACH SEGMENT IS GENERATED
  5.  DELETE /session/{id}      →  free resources

Streaming protocol
──────────────────
  Content-Type: multipart/x-mixed-replace; boundary=flashhead

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

The frontend receives and plays each part the moment it arrives.
No buffering – first video arrives after ~3 inference chunks.
"""

import asyncio
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, Optional

import imageio
import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

# ── inference backend ────────────────────────────────────────────────────────
try:
    from flash_head.inference import (
        get_audio_embedding,
        get_base_data,
        get_infer_params,
        get_pipeline,
        run_pipeline,
    )
    from flash_head.utils.facecrop import postprocess_image2
    BACKEND_OK = True
except ImportError as _e:
    BACKEND_OK = False
    logger.warning(f"flash_head not importable: {_e}")

# ── constants ─────────────────────────────────────────────────────────────────
CHUNKS_PER_SEGMENT   = 3        # inference chunks → 1 video segment  (≈3 s)
MULTIPART_BOUNDARY   = "flashhead"
SESSION_IDLE_TIMEOUT = 300      # seconds idle before auto-delete  (5 min)
SESSION_GC_INTERVAL  = 60       # how often the GC thread wakes up (seconds)


# ════════════════════════════════════════════════════════════════════════════
# Model  (singleton, one GPU)
# ════════════════════════════════════════════════════════════════════════════

class _Model:
    def __init__(self):
        self.pipeline        = None
        self.ckpt_dir        = ""
        self.wav2vec_dir     = ""
        self.model_type      = ""
        self.loaded          = False
        self.lock            = threading.Lock()

    def load(self, ckpt_dir: str, wav2vec_dir: str, model_type: str):
        with self.lock:
            logger.info(f"Loading model  ckpt={ckpt_dir}  type={model_type}")
            self.pipeline    = get_pipeline(
                world_size=1, ckpt_dir=ckpt_dir,
                wav2vec_dir=wav2vec_dir, model_type=model_type,
            )
            self.ckpt_dir    = ckpt_dir
            self.wav2vec_dir = wav2vec_dir
            self.model_type  = model_type
            self.loaded      = True
            logger.info("Model ready ✓")

MODEL = _Model()


# ════════════════════════════════════════════════════════════════════════════
# Session  (per client / conversation)
# ════════════════════════════════════════════════════════════════════════════

class Session:
    def __init__(self, sid: str):
        self.id          = sid
        self.image_path: Optional[str] = None
        self.audio_dq:   Optional[deque] = None   # persists across turns
        self.base_seed   = 9999
        self.lock        = threading.Lock()
        self.created_at  = time.time()
        self.last_used   = time.time()
        self.turn        = 0
        self._dir        = Path(tempfile.mkdtemp(prefix=f"fh_{sid[:8]}_"))

    # ── image ─────────────────────────────────────────────────────────────

    def set_image(self, data: bytes, suffix: str,
                  base_seed: int = 9999, use_face_crop: bool = True):
        path = str(self._dir / f"face{suffix}")
        with open(path, "wb") as f:
            f.write(data)
        self.image_path = path
        self.base_seed  = base_seed
        with MODEL.lock:
            get_base_data(
                MODEL.pipeline,
                cond_image_path_or_dir=path,
                base_seed=base_seed,
                use_face_crop=use_face_crop,
            )
        logger.info(f"[{self.id}] image set  face_crop={use_face_crop}")

    # ── generate (streaming) ──────────────────────────────────────────────

    def iter_segments(
        self,
        audio_data: bytes,
        audio_suffix: str,
        use_face_crop: bool,
    ):
        """
        Blocking generator.
        Yields (segment_index, mp4_bytes) as each segment finishes.
        Designed to be called from a background thread so FastAPI can
        async-iterate the results without blocking the event loop.
        """
        if self.image_path is None:
            raise RuntimeError("No image set. Call PATCH /session/{id}/image first.")

        # ── infer params ──────────────────────────────────────────────────
        p           = get_infer_params()
        sample_rate = p["sample_rate"]
        tgt_fps     = p["tgt_fps"]
        frame_num   = p["frame_num"]
        motion_num  = p["motion_frames_num"]
        cached_dur  = p["cached_audio_duration"]
        slice_len   = frame_num - motion_num

        cached_len      = sample_rate * cached_dur
        audio_end_idx   = cached_dur * tgt_fps
        audio_start_idx = audio_end_idx - frame_num
        slice_samples   = slice_len * sample_rate // tgt_fps

        # initialise (or reuse) the sliding audio context window
        if self.audio_dq is None:
            self.audio_dq = deque([0.0] * cached_len, maxlen=cached_len)

        # ── save incoming audio ───────────────────────────────────────────
        audio_path = str(self._dir / f"turn_{self.turn}{audio_suffix}")
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        speech, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

        # pad to full slice multiples
        rem = len(speech) % slice_samples
        if rem:
            speech = np.concatenate(
                [speech, np.zeros(slice_samples - rem, dtype=speech.dtype)]
            )
        slices       = speech.reshape(-1, slice_samples)
        total_chunks = len(slices)
        num_segments = (total_chunks + CHUNKS_PER_SEGMENT - 1) // CHUNKS_PER_SEGMENT

        # ── pre-save per-segment audio wavs ───────────────────────────────
        seg_audio: dict[int, str] = {}
        for sid in range(num_segments):
            start  = sid * CHUNKS_PER_SEGMENT
            end    = min(start + CHUNKS_PER_SEGMENT, total_chunks)
            concat = np.concatenate([slices[i] for i in range(start, end)])
            wav_p  = str(self._dir / f"turn_{self.turn}_seg{sid:04d}.wav")
            _write_wav(concat, wav_p, sample_rate)
            seg_audio[sid] = wav_p

        # ── inference queue ───────────────────────────────────────────────
        res_q: queue.Queue = queue.Queue()

        def inference_worker():
            audio_dq = self.audio_dq   # shared context
            with MODEL.lock:
                # make sure THIS session's image is active on the pipeline
                get_base_data(
                    MODEL.pipeline,
                    cond_image_path_or_dir=self.image_path,
                    base_seed=self.base_seed,
                    use_face_crop=use_face_crop,
                )
                for chunk_idx, speech_slice in enumerate(slices):
                    audio_dq.extend(speech_slice.tolist())
                    emb = get_audio_embedding(
                        MODEL.pipeline,
                        np.array(audio_dq),
                        audio_start_idx, audio_end_idx,
                    )
                    torch.cuda.synchronize()
                    t0    = time.time()
                    video = run_pipeline(MODEL.pipeline, emb)

                    if use_face_crop:
                        video = postprocess_image2(
                            video,
                            MODEL.pipeline.cond_image_tensor_dict["meta"]["original_image"],
                            MODEL.pipeline.cond_image_tensor_dict["meta"]["boxes"],
                        )

                    # ── this is the FIXME point in the Gradio code ────────
                    # chunk is ready → push to queue so the HTTP stream
                    # can send it to the client immediately
                    video = video[motion_num:]
                    torch.cuda.synchronize()
                    logger.info(
                        f"[{self.id}] chunk {chunk_idx}/{total_chunks-1} "
                        f"in {time.time()-t0:.2f}s"
                    )
                    res_q.put((chunk_idx, video.cpu().numpy()))

            res_q.put(None)   # sentinel

        t = threading.Thread(target=inference_worker, daemon=True)
        t.start()

        # ── main loop: collect chunks → encode segment → yield ────────────
        frame_buf: list = []

        while True:
            item = res_q.get()
            if item is None:
                break

            chunk_idx, chunk_np = item
            frame_buf.append(torch.from_numpy(chunk_np))

            if len(frame_buf) == CHUNKS_PER_SEGMENT:
                seg_idx  = (chunk_idx + 1 - CHUNKS_PER_SEGMENT) // CHUNKS_PER_SEGMENT
                mp4      = _encode_segment(frame_buf, seg_audio[seg_idx], tgt_fps)
                logger.info(
                    f"[{self.id}] segment {seg_idx} ready "
                    f"({len(mp4):,} bytes) → streaming to client"
                )
                yield seg_idx, mp4
                frame_buf = []

        # flush remainder
        if frame_buf:
            seg_idx = num_segments - 1
            mp4     = _encode_segment(frame_buf, seg_audio[seg_idx], tgt_fps)
            logger.info(
                f"[{self.id}] final segment {seg_idx} "
                f"({len(mp4):,} bytes) → streaming to client"
            )
            yield seg_idx, mp4

        t.join()
        self.turn     += 1
        self.last_used = time.time()

    # ── cleanup ───────────────────────────────────────────────────────────

    def destroy(self):
        shutil.rmtree(self._dir, ignore_errors=True)
        logger.info(f"[{self.id}] session destroyed")


# ── registry ──────────────────────────────────────────────────────────────────
_sessions: dict[str, Session] = {}
_reg_lock  = threading.Lock()

def _get(sid: str) -> Session:
    with _reg_lock:
        s = _sessions.get(sid)
    if not s:
        raise HTTPException(404, f"Session '{sid}' not found.")
    return s


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _write_wav(arr: np.ndarray, path: str, sr: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def _encode_segment(frames: list, audio_path: str, fps: int) -> bytes:
    """Encode frame tensors + audio → MP4 bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        silent = os.path.join(tmp, "s.mp4")
        final  = os.path.join(tmp, "f.mp4")

        with imageio.get_writer(
            silent, format="mp4", mode="I", fps=fps,
            codec="h264", ffmpeg_params=["-bf", "0"],
        ) as w:
            for f in frames:
                arr = f.numpy().astype(np.uint8)
                for i in range(arr.shape[0]):
                    w.append_data(arr[i])

        subprocess.run(
            ["ffmpeg", "-y",
             "-i", silent, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac",
             final],
            check=True, capture_output=True,
        )
        return Path(final).read_bytes()


def _multipart_stream(
    loop: asyncio.AbstractEventLoop,
    seg_iter,          # blocking generator from session.iter_segments()
) -> AsyncGenerator[bytes, None]:
    """
    Bridges the blocking segment generator to an async generator
    by running each .next() call in the thread-pool executor.
    Yields raw multipart bytes.
    """
    seg_iter_obj  = iter(seg_iter)
    _sentinel     = object()

    async def gen():
        while True:
            item = await loop.run_in_executor(
                None, lambda: next(seg_iter_obj, _sentinel)
            )
            if item is _sentinel:
                break

            seg_idx, mp4_bytes = item
            header = (
                f"--{MULTIPART_BOUNDARY}\r\n"
                f"Content-Type: video/mp4\r\n"
                f"X-Segment-Index: {seg_idx}\r\n"
                f"Content-Length: {len(mp4_bytes)}\r\n"
                f"\r\n"
            ).encode()
            yield header
            yield mp4_bytes
            yield b"\r\n"

        yield f"--{MULTIPART_BOUNDARY}--\r\n".encode()

    return gen()


# ════════════════════════════════════════════════════════════════════════════
# Session garbage collector  (background daemon thread)
# ════════════════════════════════════════════════════════════════════════════

def _session_gc():
    """
    Wakes every SESSION_GC_INTERVAL seconds.
    Destroys any session idle longer than SESSION_IDLE_TIMEOUT seconds.
    """
    while True:
        time.sleep(SESSION_GC_INTERVAL)
        now     = time.time()
        expired = []

        with _reg_lock:
            for sid, s in list(_sessions.items()):
                if (now - s.last_used) > SESSION_IDLE_TIMEOUT:
                    expired.append(sid)

        for sid in expired:
            with _reg_lock:
                s = _sessions.pop(sid, None)
            if s:
                s.destroy()
                logger.info(
                    f"[GC] auto-deleted session {sid}  "
                    f"(idle > {SESSION_IDLE_TIMEOUT}s)"
                )

        if expired:
            logger.info(f"[GC] swept {len(expired)} idle session(s).")


# ════════════════════════════════════════════════════════════════════════════
# FastAPI  (lifespan starts GC on boot)
# ════════════════════════════════════════════════════════════════════════════

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_app):
    t = threading.Thread(target=_session_gc, daemon=True, name="session-gc")
    t.start()
    logger.info(
        f"Session GC started — "
        f"idle timeout {SESSION_IDLE_TIMEOUT}s, "
        f"check interval {SESSION_GC_INTERVAL}s"
    )
    yield   # server runs here


app = FastAPI(
    title="FlashHead Streaming API",
    description=(
        "POST audio → receive MP4 video segments streamed back chunk by chunk.\n\n"
        "See `POST /session/{id}/generate` for the streaming endpoint."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── model ─────────────────────────────────────────────────────────────────────

@app.post("/model/load", tags=["Model"])
def model_load(
    ckpt_dir:    str = Form(...),
    wav2vec_dir: str = Form(...),
    model_type:  str = Form("lite"),
):
    """Load FlashHead + Wav2Vec2 into GPU memory."""
    if not BACKEND_OK:
        raise HTTPException(501, "flash_head package not installed.")
    if model_type not in ("lite", "pro"):
        raise HTTPException(422, "model_type must be 'lite' or 'pro'.")
    try:
        MODEL.load(ckpt_dir, wav2vec_dir, model_type)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"status": "loaded", "model_type": model_type}


@app.get("/model/status", tags=["Model"])
def model_status():
    return {
        "loaded":      MODEL.loaded,
        "model_type":  MODEL.model_type,
        "ckpt_dir":    MODEL.ckpt_dir,
        "wav2vec_dir": MODEL.wav2vec_dir,
    }


# ── session ───────────────────────────────────────────────────────────────────

@app.post("/session", tags=["Session"])
async def session_create(
    image:         UploadFile = File(..., description="Face/portrait image"),
    base_seed:     int        = Form(9999),
    use_face_crop: bool       = Form(True),
):
    """Create a session and set the condition image."""
    if not MODEL.loaded:
        raise HTTPException(400, "Model not loaded. Call POST /model/load first.")

    sid     = str(uuid.uuid4())
    session = Session(sid)
    data    = await image.read()
    suffix  = Path(image.filename).suffix or ".png"
    try:
        session.set_image(data, suffix, base_seed, use_face_crop)
    except Exception as e:
        session.destroy()
        raise HTTPException(500, str(e))

    with _reg_lock:
        _sessions[sid] = session

    return {"session_id": sid, "status": "ready"}


@app.patch("/session/{session_id}/image", tags=["Session"])
async def session_update_image(
    session_id:    str,
    image:         UploadFile = File(...),
    base_seed:     int        = Form(9999),
    use_face_crop: bool       = Form(True),
):
    """Hot-swap the face image (e.g. the caller changed)."""
    s    = _get(session_id)
    data = await image.read()
    suffix = Path(image.filename).suffix or ".png"
    try:
        with s.lock:
            s.set_image(data, suffix, base_seed, use_face_crop)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"session_id": session_id, "status": "image_updated"}


@app.post(
    "/session/{session_id}/generate",
    tags=["Generate"],
    summary="Send audio → receive MP4 video segments streamed back in real time",
    responses={
        200: {
            "content": {
                f"multipart/x-mixed-replace; boundary={MULTIPART_BOUNDARY}": {}
            },
            "description": (
                "Multipart stream. Each part is a complete MP4 segment "
                "sent as soon as it is generated. "
                "Headers per part: Content-Type, X-Segment-Index, Content-Length."
            ),
        }
    },
)
async def session_generate(
    session_id:    str,
    audio:         UploadFile = File(..., description="Audio (.wav 16 kHz mono recommended)"),
    use_face_crop: bool       = Form(True),
):
    """
    **This is the main streaming endpoint.**

    - Send the audio for one chatbot turn.
    - The response body is a `multipart/x-mixed-replace` stream.
    - Parts arrive **as each ~3-second video segment is generated** —
      no waiting for the full video.
    - Read the stream continuously and play / display each MP4 part
      as it arrives.
    """
    session     = _get(session_id)
    audio_bytes = await audio.read()
    suffix      = Path(audio.filename).suffix or ".wav"

    loop = asyncio.get_event_loop()

    # Build the blocking segment generator (runs inference in a thread)
    try:
        seg_gen = session.iter_segments(audio_bytes, suffix, use_face_crop)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

    return StreamingResponse(
        _multipart_stream(loop, seg_gen),
        media_type=f"multipart/x-mixed-replace; boundary={MULTIPART_BOUNDARY}",
        headers={"X-Session-ID": session_id},
    )


@app.get("/session/{session_id}/status", tags=["Session"])
def session_status(session_id: str):
    s = _get(session_id)
    return {
        "session_id":   s.id,
        "has_image":    s.image_path is not None,
        "turn":         s.turn,
        "created_at":   s.created_at,
        "last_used":    s.last_used,
        "idle_seconds": round(time.time() - s.last_used, 1),
    }


@app.delete("/session/{session_id}", tags=["Session"])
def session_delete(session_id: str):
    s = _get(session_id)
    s.destroy()
    with _reg_lock:
        _sessions.pop(session_id, None)
    return {"session_id": session_id, "status": "deleted"}


@app.get("/sessions", tags=["Session"])
def sessions_list():
    with _reg_lock:
        rows = list(_sessions.values())
    return {
        "count": len(rows),
        "sessions": [
            {"session_id": s.id, "turn": s.turn,
             "idle_seconds": round(time.time() - s.last_used, 1)}
            for s in rows
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser("FlashHead Streaming API Server")
    parser.add_argument("--host",            default="0.0.0.0")
    parser.add_argument("--port",            default=8000, type=int)
    parser.add_argument("--ckpt-dir",        default=None, help="Auto-load model on start")
    parser.add_argument("--wav2vec-dir",     default=None)
    parser.add_argument("--model-type",      default="lite", choices=["lite", "pro"])
    parser.add_argument("--idle-timeout",    default=300, type=int,
                        help="Seconds before an idle session is auto-deleted (default 300)")
    parser.add_argument("--gc-interval",     default=60,  type=int,
                        help="How often the GC check runs in seconds (default 60)")
    args = parser.parse_args()

    # apply GC config before server starts
    SESSION_IDLE_TIMEOUT = args.idle_timeout
    SESSION_GC_INTERVAL  = args.gc_interval

    if args.ckpt_dir and args.wav2vec_dir:
        MODEL.load(args.ckpt_dir, args.wav2vec_dir, args.model_type)

    uvicorn.run(app, host=args.host, port=args.port)