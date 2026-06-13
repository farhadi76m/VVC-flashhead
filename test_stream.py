"""
FlashHead · Stream Tester
══════════════════════════════════════════════════════════════════════
Usage
─────
  # Live stream: segments arrive, each one is appended to output.mp4
  python test_stream.py live first_frame.png example.wav

  # Split an already-downloaded .bin file into individual MP4s
  python test_stream.py split o.bin

Live mode behaviour
───────────────────
  [ 0.1s]  session created
  [ 0.2s]  stream open — waiting for segments …
  [ 8.3s]  ▶ segment 0  (512,048 bytes) → appended → output.mp4  [1 seg]
  [13.1s]  ▶ segment 1  (498,112 bytes) → appended → output.mp4  [2 segs]
  [17.9s]  ▶ segment 2  (501,760 bytes) → appended → output.mp4  [3 segs]
  [18.0s]  ✅ done — output.mp4  (total 3 segments, 47.9 s video, 1.4 MB)
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

BASE     = "http://localhost:8000"
BOUNDARY = b"flashhead"


# ════════════════════════════════════════════════════════════════════════════
# Multipart stream parser  (same as client_example.py)
# ════════════════════════════════════════════════════════════════════════════

def iter_segments(response: requests.Response):
    """Yield (seg_idx, mp4_bytes) as each multipart part arrives."""
    buf      = b""
    in_body  = False
    body_len = 0
    seg_idx  = 0

    for raw in response.iter_content(chunk_size=8192):
        buf += raw
        while True:
            if not in_body:
                end = buf.find(b"\r\n\r\n")
                if end == -1:
                    break
                headers = buf[:end].decode(errors="replace")
                buf     = buf[end + 4:]
                m = re.search(r"Content-Length:\s*(\d+)", headers, re.I)
                if not m:
                    break
                body_len = int(m.group(1))
                m2 = re.search(r"X-Segment-Index:\s*(\d+)", headers, re.I)
                if m2:
                    seg_idx = int(m2.group(1))
                in_body = True
            else:
                if len(buf) < body_len:
                    break
                yield seg_idx, buf[:body_len]
                buf     = buf[body_len:]
                in_body = False
                seg_idx += 1
                if buf.startswith(b"\r\n"):
                    buf = buf[2:]


# ════════════════════════════════════════════════════════════════════════════
# ffmpeg helpers
# ════════════════════════════════════════════════════════════════════════════

def _concat_segments(seg_paths: list[str], out_path: str):
    """
    Concatenate all seg_paths into out_path using ffmpeg concat demuxer.
    Called after every new segment so output.mp4 is always up-to-date.
    """
    # write a concat list file
    list_file = out_path + ".concat_list.txt"
    with open(list_file, "w") as f:
        for p in seg_paths:
            f.write(f"file '{Path(p).resolve()}'\n")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",          # no re-encode, just remux
            out_path,
        ],
        check=True,
        capture_output=True,
    )
    os.remove(list_file)


def _video_duration(path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# ════════════════════════════════════════════════════════════════════════════
# Live streaming test
# ════════════════════════════════════════════════════════════════════════════

def stream_live(
    image_path: str,
    audio_path: str,
    output_video: str = "output.mp4",
):
    t0      = time.time()
    tmp_dir = tempfile.mkdtemp(prefix="fh_test_")
    seg_files: list[str] = []

    def elapsed():
        return f"[{time.time()-t0:5.1f}s]"

    # ── create session ────────────────────────────────────────────────────
    with open(image_path, "rb") as f:
        r = requests.post(f"{BASE}/session", files={"image": f})
    r.raise_for_status()
    sid = r.json()["session_id"]
    print(f"{elapsed()}  session  {sid}")

    # ── open streaming request ────────────────────────────────────────────
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{BASE}/session/{sid}/generate",
            files={"audio": f},
            stream=True,
        )
    resp.raise_for_status()
    print(f"{elapsed()}  stream open — waiting for segments …")
    print()

    # ── consume stream: save each segment then re-concat ─────────────────
    for seg_idx, mp4_bytes in iter_segments(resp):
        # 1. save segment to temp file
        seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:04d}.mp4")
        with open(seg_path, "wb") as f:
            f.write(mp4_bytes)
        seg_files.append(seg_path)

        # 2. append to output video (ffmpeg concat, no re-encode)
        _concat_segments(seg_files, output_video)

        # 3. report
        size_mb  = Path(output_video).stat().st_size / 1024 / 1024
        dur      = _video_duration(output_video)
        print(
            f"{elapsed()}  ▶ segment {seg_idx:3d}  "
            f"({len(mp4_bytes):>9,} bytes)  →  appended  "
            f"→  {output_video}  "
            f"[{len(seg_files)} seg{'s' if len(seg_files)>1 else ''}  "
            f"{dur:.1f}s  {size_mb:.1f} MB]"
        )

    # ── final summary ─────────────────────────────────────────────────────
    print()
    if seg_files:
        size_mb = Path(output_video).stat().st_size / 1024 / 1024
        dur     = _video_duration(output_video)
        print(
            f"{elapsed()}  ✅  done\n"
            f"             file    : {output_video}\n"
            f"             segments: {len(seg_files)}\n"
            f"             duration: {dur:.1f}s\n"
            f"             size    : {size_mb:.2f} MB"
        )
    else:
        print(f"{elapsed()}  ⚠  no segments received")

    # ── cleanup ───────────────────────────────────────────────────────────
    requests.delete(f"{BASE}/session/{sid}")
    for p in seg_files:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass


# ════════════════════════════════════════════════════════════════════════════
# Split an existing .bin file
# ════════════════════════════════════════════════════════════════════════════

def split_bin(bin_path: str = "o.bin", output_video: str = "output.mp4"):
    data  = open(bin_path, "rb").read()
    delim = b"--" + BOUNDARY
    parts = data.split(delim)

    tmp_dir   = tempfile.mkdtemp(prefix="fh_split_")
    seg_files = []

    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        sep = part.find(b"\r\n\r\n")
        if sep == -1:
            continue
        headers_raw = part[:sep].decode(errors="replace")
        body        = part[sep + 4:]
        if body.endswith(b"\r\n"):
            body = body[:-2]

        m   = re.search(r"X-Segment-Index:\s*(\d+)", headers_raw, re.I)
        idx = int(m.group(1)) if m else len(seg_files)

        seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")
        with open(seg_path, "wb") as f:
            f.write(body)
        seg_files.append(seg_path)
        print(f"  extracted segment {idx:3d}  ({len(body):,} bytes)")

    if not seg_files:
        print("No segments found.")
        return

    print(f"\nConcatenating {len(seg_files)} segments → {output_video} …")
    _concat_segments(seg_files, output_video)
    size_mb = Path(output_video).stat().st_size / 1024 / 1024
    dur     = _video_duration(output_video)
    print(f"✅  {output_video}  ({dur:.1f}s  {size_mb:.2f} MB)")

    for p in seg_files:
        os.remove(p)
    os.rmdir(tmp_dir)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "live":
        # python test_stream.py live <image> <audio> [output.mp4]
        if len(sys.argv) < 4:
            print("usage: python test_stream.py live <image.png> <audio.wav> [output.mp4]")
            sys.exit(1)
        out = sys.argv[4] if len(sys.argv) > 4 else "output.mp4"
        stream_live(sys.argv[2], sys.argv[3], out)

    elif cmd == "split":
        # python test_stream.py split [file.bin] [output.mp4]
        bin_file = sys.argv[2] if len(sys.argv) > 2 else "o.bin"
        out      = sys.argv[3] if len(sys.argv) > 3 else "output.mp4"
        split_bin(bin_file, out)

    else:
        print(__doc__)