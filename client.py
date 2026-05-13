"""
FlashHead API · Client Examples
════════════════════════════════
Shows how to consume the multipart streaming response.
"""

import re
import requests

BASE = "http://localhost:8002"
BOUNDARY = b"flashhead"


# ════════════════════════════════════════════════════════════════════════════
# Multipart stream parser
# ════════════════════════════════════════════════════════════════════════════

def iter_mp4_chunks(response: requests.Response):
    """
    Parse the multipart/x-mixed-replace stream from /generate.
    Yields (segment_index, mp4_bytes) for each part as it arrives.
    """
    buf        = b""
    in_body    = False
    seg_index  = 0
    body_len   = 0

    for raw_chunk in response.iter_content(chunk_size=8192):
        buf += raw_chunk

        while True:
            if not in_body:
                # look for end of part headers
                header_end = buf.find(b"\r\n\r\n")
                if header_end == -1:
                    break

                headers_raw = buf[:header_end].decode(errors="replace")
                buf         = buf[header_end + 4:]

                # parse Content-Length and X-Segment-Index
                m = re.search(r"Content-Length:\s*(\d+)", headers_raw, re.I)
                if not m:
                    break
                body_len  = int(m.group(1))

                m2 = re.search(r"X-Segment-Index:\s*(\d+)", headers_raw, re.I)
                seg_index = int(m2.group(1)) if m2 else seg_index

                in_body = True

            else:
                # collect exactly body_len bytes
                if len(buf) < body_len:
                    break   # need more data

                mp4_bytes = buf[:body_len]
                buf       = buf[body_len:]
                in_body   = False

                yield seg_index, mp4_bytes
                seg_index += 1

                # skip \r\n after body
                if buf.startswith(b"\r\n"):
                    buf = buf[2:]


# ════════════════════════════════════════════════════════════════════════════
# Simple client
# ════════════════════════════════════════════════════════════════════════════

class FlashHeadClient:

    def __init__(self, base: str = BASE):
        self.base       = base.rstrip("/")
        self.session_id = None

    # ── model ──────────────────────────────────────────────────────────────

    def load_model(
        self,
        ckpt_dir    = "models/SoulX-FlashHead-1_3B",
        wav2vec_dir = "models/wav2vec2-base-960h",
        model_type  = "lite",
    ):
        r = requests.post(f"{self.base}/model/load", data=dict(
            ckpt_dir=ckpt_dir, wav2vec_dir=wav2vec_dir, model_type=model_type,
        ))
        r.raise_for_status()
        return r.json()

    # ── session ────────────────────────────────────────────────────────────

    def create_session(self, image_path: str, use_face_crop: bool = True) -> str:
        with open(image_path, "rb") as f:
            r = requests.post(
                f"{self.base}/session",
                files={"image": f},
                data={"use_face_crop": use_face_crop},
            )
        r.raise_for_status()
        self.session_id = r.json()["session_id"]
        print(f"Session: {self.session_id}")
        return self.session_id

    def update_image(self, image_path: str, use_face_crop: bool = True):
        with open(image_path, "rb") as f:
            r = requests.patch(
                f"{self.base}/session/{self.session_id}/image",
                files={"image": f},
                data={"use_face_crop": use_face_crop},
            )
        r.raise_for_status()
        return r.json()

    # ── generate (streaming) ───────────────────────────────────────────────

    def generate_stream(self, audio_path: str, output_prefix: str = "seg"):
        """
        Send audio, receive MP4 segments as they are generated.
        Saves each segment as {output_prefix}_{idx:04d}.mp4
        Returns list of saved paths.
        """
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{self.base}/session/{self.session_id}/generate",
                files={"audio": f},
                stream=True,   # ← key: don't buffer the whole response
            )
        breakpoint()    
        response.raise_for_status()

        saved = []
        for seg_idx, mp4_bytes in iter_mp4_chunks(response):
            path = f"{output_prefix}_{seg_idx:04d}.mp4"
            with open(path, "wb") as out:
                out.write(mp4_bytes)
            print(f"  segment {seg_idx:4d}  {len(mp4_bytes):>10,} bytes  → {path}")
            saved.append(path)

        return saved

    def delete_session(self):
        r = requests.delete(f"{self.base}/session/{self.session_id}")
        self.session_id = None
        return r.json()


# ════════════════════════════════════════════════════════════════════════════
# Demo
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    client = FlashHeadClient()

    print("1. Loading model …")
    print(client.load_model())

    print("\n2. Creating session with face image …")
    client.create_session("examples/first_frame.png", use_face_crop=True)

    print("\n3. Turn 1 – generating video from audio …")
    segs = client.generate_stream("examples/example.wav", output_prefix="turn1_seg")
    print(f"   Got {len(segs)} segments")

    print("\n4. Turn 2 – audio context (deque) carries over …")
    segs = client.generate_stream("examples/podcast_sichuan_16k.wav", output_prefix="turn2_seg")
    print(f"   Got {len(segs)} segments")

    print("\n5. Caller changed face – hot-swap image …")
    client.update_image("examples/girl.png", use_face_crop=True)

    print("\n6. Turn 3 – same session, new face …")
    segs = client.generate_stream("examples/example.wav", output_prefix="turn3_seg")
    print(f"   Got {len(segs)} segments")

    print("\n7. Cleanup …")
    print(client.delete_session())


# ════════════════════════════════════════════════════════════════════════════
# cURL examples
# ════════════════════════════════════════════════════════════════════════════
"""
# Start server (model loads automatically)
python server.py \
  --ckpt-dir  models/SoulX-FlashHead-1_3B \
  --wav2vec-dir models/wav2vec2-base-960h \
  --model-type lite

# 1. Create session
SESSION=$(curl -s -X POST http://localhost:8000/session \
  -F "image=@examples/girl.png" \
  -F "use_face_crop=true" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

echo "Session: $SESSION"

# 2. Generate – segments stream in as they are generated
#    curl writes each received byte to the file in real time
curl -X POST "http://localhost:8000/session/$SESSION/generate" \
  -F "audio=@examples/speech.wav" \
  --output stream_output.bin

# 3. Hot-swap image
curl -X PATCH "http://localhost:8000/session/$SESSION/image" \
  -F "image=@examples/new_face.png"

# 4. Next turn
curl -X POST "http://localhost:8000/session/$SESSION/generate" \
  -F "audio=@examples/speech2.wav" \
  --output stream_output2.bin

# 5. Delete
curl -X DELETE "http://localhost:8000/session/$SESSION"

# Interactive docs
open http://localhost:8000/docs
"""
