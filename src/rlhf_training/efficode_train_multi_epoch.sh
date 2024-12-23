# !/bin/bash


# pretrain_path="codellama/CodeLlama-7b-Instruct-hf"
# base_save_path="./checkpoints_efficiency_both/codellama/CodeLlama-7b-Instruct-hf-epoch"
pretrain_path="Qwen/CodeQwen1.5-7B-Chat"
base_save_path="./checkpoints_efficiency_both/CodeQwen/CodeQwen1.5-7B-Chat"
# pretrain_path="ise-uiuc/Magicoder-S-DS-6.7B"
# base_save_path="./checkpoints_efficiency_both/magicoder/magicoder-6.7B-epoch"
# pretrain_path="deepseek-ai/deepseek-coder-6.7b-instruct" 
# base_save_path="./checkpoints_efficiency_both/deepseek/deepseek-coder-6.7b-instruct-epoch"
# pretrain_path="microsoft/wavecoder-ultra-6.7b"
# base_save_path="./checkpoints_efficiency_both/wavecoder/wavecoder-ultra-6.7b-epoch"

start_iteration=1
max_iteration=20 # You can modify this to your desired maximum
iteration=$start_iteration

while [ $iteration -le $max_iteration ]; do
  save_path="${base_save_path}-${iteration}"
  echo "Starting iteration ${iteration}..."

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --include localhost:2,3 --master_port=29503 OpenRLHF/openrlhf/cli/train_ppo.py \
    --pretrain $pretrain_path \
    --save_path $save_path \
    --logging_steps 1 \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 512 \
    --max_epochs $iteration \
    --prompt_max_len 1500 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --input_key prompt \
    --max_samples 100000 \
    --flash_attn \
    --gradient_checkpointing \
    --remote_rm_url http://localhost:5000/get_reward \
    --prompt_data json@/workspace/openrhlf_test/data_chat_template \
    --max_ckpt_num 10 \
    --eval_steps -1 \
    --n_samples_per_prompt 5 \
    --adam_offload \
    --apply_chat_template 

  echo "Iteration ${iteration} complete. Model saved at ${save_path}"
  
  # Increment the iteration by 5
  iteration=$((iteration + 2))
done


# !/bin/bash


# # pretrain_path="codellama/CodeLlama-7b-Instruct-hf"
# # base_save_path="./checkpoints/codellama/CodeLlama-7b-Instruct-hf-epoch"
# pretrain_path="Qwen/CodeQwen1.5-7B-Chat"
# base_save_path="./checkpoints/CodeQwen/CodeQwen1.5-7B-Chat"
# # pretrain_path="ise-uiuc/Magicoder-S-DS-6.7B"
# # base_save_path="./checkpoints/magicoder/magicoder-6.7B-epoch"
# # pretrain_path="deepseek-ai/deepseek-coder-6.7b-instruct"
# # base_save_path="./checkpoints/deepseek/deepseek-coder-6.7b-instruct-epoch"

# start_iteration=3
# max_iteration=30 # You can modify this to your desired maximum
# iteration=$start_iteration

# while [ $iteration -le $max_iteration ]; do
#   save_path="${base_save_path}-${iteration}"
#   echo "Starting iteration ${iteration}..."

#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --include localhost:6,7 --master_port=29504 /workspace/openrhlf_test/OpenRLHF/openrlhf/cli/train_ppo.py \
#     --pretrain $pretrain_path \
#     --save_path $save_path \
#     --logging_steps 1 \
#     --micro_train_batch_size 4 \
#     --train_batch_size 128 \
#     --micro_rollout_batch_size 8 \
#     --rollout_batch_size 512 \
#     --max_epochs $iteration \
#     --prompt_max_len 1500 \
#     --generate_max_len 1024 \
#     --zero_stage 2 \
#     --bf16 \
#     --actor_learning_rate 5e-7 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.01 \
#     --input_key prompt \
#     --max_samples 100000 \
#     --flash_attn \
#     --gradient_checkpointing \
#     --remote_rm_url http://localhost:5000/get_reward \
#     --prompt_data json@/workspace/openrhlf_test/data_chat_template \
#     --max_ckpt_num 10 \
#     --eval_steps -1 \
#     --n_samples_per_prompt 5 \
#     --adam_offload \
#     --apply_chat_template 

#   echo "Iteration ${iteration} complete. Model saved at ${save_path}"
  
#   # Increment the iteration by 5
#   iteration=$((iteration + 3))
# done


