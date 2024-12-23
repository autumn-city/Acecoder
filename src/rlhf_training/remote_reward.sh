CUDA_VISIBLE_DEVICES=6 python -m remote_reward_v3 \
    --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
    --port 5000 \
    --bf16 \
    --flash_attn \
    --max_len 8192 \
    --batch_size 16 \
    --power 1 \