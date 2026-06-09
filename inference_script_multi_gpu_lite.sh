CUDA_VISIBLE_DEVICES=0,1
GPU_NUM=2
export NCCL_MIN_NCHANNELS=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=$GPU_NUM generate_video.py \
    --ckpt_dir models/SoulX-FlashHead-1_3B \
    --wav2vec_dir models/wav2vec2-base-960h \
    --model_type lite \
    --cond_image examples/erfan.jpg \
    --audio_path examples/new_sound.wav \
    --audio_encode_mode stream