"""
Gradio streaming video generation: inference and saving run asynchronously for real-time output.
"""
import gradio as gr
import os
import torch
import numpy as np
import wave
import imageio
import librosa
import subprocess
import queue
import threading
from datetime import datetime
from collections import deque

from flash_head.inference import (
    get_pipeline,
    get_base_data,
    get_infer_params,
    get_audio_embedding,
    run_pipeline,
)

CHUNKS_PER_SEGMENT = 3

pipeline = None
loaded_ckpt_dir = None
loaded_wav2vec_dir = None
loaded_model_type = None


def _write_frames_to_mp4(frames_list, video_path, fps):
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    with imageio.get_writer(
        video_path, format="mp4", mode="I", fps=fps,
        codec="h264", ffmpeg_params=["-bf", "0"],
    ) as writer:
        for frames in frames_list:
            frames_np = frames.numpy().astype(np.uint8)
            for i in range(frames_np.shape[0]):
                writer.append_data(frames_np[i])
    return video_path


def save_video_with_audio(frames_list, video_path, audio_path, fps):
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    _write_frames_to_mp4(frames_list, temp_path, fps)
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_path,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac",
            video_path,
        ], check=True, capture_output=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return video_path


def _save_chunk_audio_to_wav(audio_array, wav_path, sample_rate=16000):
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    samples = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())
    return wav_path


def run_inference_streaming(
    ckpt_dir, wav2vec_dir, model_type,
    cond_image, audio_path, seed, use_face_crop,
    progress=gr.Progress(),
):
    global pipeline, loaded_ckpt_dir, loaded_wav2vec_dir, loaded_model_type

    if (pipeline is None or loaded_ckpt_dir != ckpt_dir
            or loaded_wav2vec_dir != wav2vec_dir or loaded_model_type != model_type):
        progress(0.2, desc="Loading Model...")
        try:
            pipeline = get_pipeline(
                world_size=1, ckpt_dir=ckpt_dir,
                model_type=model_type, wav2vec_dir=wav2vec_dir,
            )
            loaded_ckpt_dir, loaded_wav2vec_dir, loaded_model_type = ckpt_dir, wav2vec_dir, model_type
        except Exception as e:
            raise gr.Error(f"Failed to load model: {e}")

    progress(0.5, desc="Preparing Data...")
    base_seed = int(seed) if seed >= 0 else 9999
    try:
        get_base_data(pipeline, cond_image_path_or_dir=cond_image,
                      base_seed=base_seed, use_face_crop=use_face_crop)
    except Exception as e:
        raise gr.Error(f"Error processing inputs: {e}")

    infer_params = get_infer_params()
    sample_rate = infer_params["sample_rate"]
    tgt_fps = infer_params["tgt_fps"]
    cached_audio_duration = infer_params["cached_audio_duration"]
    frame_num = infer_params["frame_num"]
    motion_frames_num = infer_params["motion_frames_num"]
    slice_len = frame_num - motion_frames_num

    try:
        human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise gr.Error(f"Failed to load audio: {e}")

    slice_samples = slice_len * sample_rate // tgt_fps
    stream_dir = os.path.join("gradio_results", "stream_preview")
    os.makedirs(stream_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    accumulated = []

    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    remainder = len(human_speech_array_all) % slice_samples
    if remainder:
        human_speech_array_all = np.concatenate([
            human_speech_array_all,
            np.zeros(slice_samples - remainder, dtype=human_speech_array_all.dtype)
        ])
    human_speech_array_slices = human_speech_array_all.reshape(-1, slice_samples)
    total_chunks = len(human_speech_array_slices)

    if total_chunks == 0:
        raise gr.Error("Audio too short. Please use a longer audio.")

    # Pre-save segment audio files
    num_segments = (total_chunks + CHUNKS_PER_SEGMENT - 1) // CHUNKS_PER_SEGMENT
    segment_audio_paths = {}
    for seg_id in range(num_segments):
        start, end = seg_id * CHUNKS_PER_SEGMENT, min(seg_id * CHUNKS_PER_SEGMENT + CHUNKS_PER_SEGMENT, total_chunks)
        audio_concat = np.concatenate([human_speech_array_slices[i] for i in range(start, end)])
        path = os.path.join(stream_dir, f"audio_{timestamp}_seg_{seg_id:04d}.wav")
        _save_chunk_audio_to_wav(audio_concat, path, sample_rate)
        segment_audio_paths[seg_id] = path

    res_queue = queue.Queue()

    def inference_worker():
        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
        for chunk_idx, speech_slice in enumerate(human_speech_array_slices):
            audio_dq.extend(speech_slice.tolist())
            embedding = get_audio_embedding(pipeline, np.array(audio_dq), audio_start_idx, audio_end_idx)
            torch.cuda.synchronize()
            video = run_pipeline(pipeline, embedding)
            video = video[motion_frames_num:]
            torch.cuda.synchronize()
            res_queue.put((chunk_idx, video.cpu().numpy()))
        res_queue.put(None)

    threading.Thread(target=inference_worker).start()

    frame_buffer = []
    while True:
        item = res_queue.get()
        if item is None:
            break
        chunk_idx, chunk_np = item
        chunk_frames = torch.from_numpy(chunk_np)
        accumulated.append(chunk_frames)
        frame_buffer.append(chunk_frames)

        if len(frame_buffer) == CHUNKS_PER_SEGMENT:
            seg_id = (chunk_idx + 1 - CHUNKS_PER_SEGMENT) // CHUNKS_PER_SEGMENT
            seg_path = os.path.join(stream_dir, f"preview_{timestamp}_seg_{seg_id:04d}.mp4")
            save_video_with_audio(frame_buffer, seg_path, segment_audio_paths[seg_id], tgt_fps)
            yield os.path.abspath(seg_path)
            frame_buffer = []

    if frame_buffer:
        seg_id = num_segments - 1
        seg_path = os.path.join(stream_dir, f"preview_{timestamp}_seg_{seg_id:04d}.mp4")
        save_video_with_audio(frame_buffer, seg_path, segment_audio_paths[seg_id], tgt_fps)
        yield os.path.abspath(seg_path)

    if not accumulated:
        raise gr.Error("No frames generated. Check inputs and retry.")

    final_path = os.path.join("gradio_results", f"res_{timestamp}.mp4")
    os.makedirs("gradio_results", exist_ok=True)
    save_video_with_audio(accumulated, final_path, audio_path, tgt_fps)


# ---------- Gradio UI ----------
with gr.Blocks(title="SoulX-FlashHead Streaming", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚡ SoulX-FlashHead Streaming Video Generation")
    gr.Markdown("Upload an image and audio — generates and plays in real time. Single GPU only.")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎬 Inputs")
                with gr.Row():
                    cond_image_input = gr.Image(label="Condition Image", type="filepath",
                                                value="examples/girl.png", height=300)
                    audio_path_input = gr.Audio(label="Audio Input", type="filepath",
                                                value="examples/podcast_sichuan_16k.wav")
            generate_btn = gr.Button("🚀 Generate (Streaming)", variant="primary", size="lg")
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                ckpt_dir_input = gr.Textbox(label="Checkpoint Directory", value="models/SoulX-FlashHead-1_3B")
                wav2vec_dir_input = gr.Textbox(label="Wav2Vec Directory", value="models/wav2vec2-base-960h")
                model_type_input = gr.Dropdown(label="Model Type", choices=["pro", "lite"], value="lite")
                use_face_crop_input = gr.Checkbox(label="Use Face Crop", value=False)
                seed_input = gr.Number(label="Random Seed", value=9999, precision=0)
        with gr.Column(scale=1):
            gr.Markdown("### 📺 Output (Streaming)")
            video_output = gr.Video(label="Generated Video", height=512,
                                    format="mp4", streaming=True, autoplay=True)

    generate_btn.click(
        fn=run_inference_streaming,
        inputs=[ckpt_dir_input, wav2vec_dir_input, model_type_input,
                cond_image_input, audio_path_input, seed_input, use_face_crop_input],
        outputs=video_output,
    )

if __name__ == "__main__":
    app.launch()