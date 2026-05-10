CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python generate_video.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B \
    --wav2vec_dir models/wav2vec2-base-960h \
    --model_type lite \
    --cond_image examples/first_frame.png \
    --audio_path examples/example.wav \
    --audio_encode_mode stream \
    --use_face_crop True