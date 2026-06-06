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

"""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger


# ════════════════════════════════════════════════════════════════════════════
# FastAPI
# ════════════════════════════════════════════════════════════════════════════


app = FastAPI(
    title="FlashHead Streaming API",
    description=(
        "POST audio → receive MP4 video segments streamed back chunk by chunk.\n\n"
        "See `POST /session/{id}/generate` for the streaming endpoint."
    ),
    version="2.0.0",
)


# model 

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
