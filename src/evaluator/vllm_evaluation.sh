model_path="/workspace/openrhlf_test/checkpoints/magicoder/magicoder-6.7B-epoch"

num_iterations=15
start_iteration=9
max_iteration=30 # You can modify this to your desired maximum
iteration=$start_iteration

while [ $iteration -le $max_iteration ]; do
    save_path="${model_path}-${iteration}"
    CUDA_VISIBLE_DEVICES=4,5,6,7 python vllm_evaluation.py \
        --checkpoint $save_path \
        --number_of_sequences 3 \
        --batch_size 8 \
        --number_0f_GPUs 4 \


    CUDA_VISIBLE_DEVICES=4,5,6,7 python vllm_evaluation.py \
        --checkpoint 'ise-uiuc/Magicoder-S-DS-6.7B' \
        --number_of_sequences 3 \
        --batch_size 8 \
        --number_0f_GPUs 4 \

    python execution_time_statistic.py \
        --ref_model 'ise-uiuc/Magicoder-S-DS-6.7B' \
        --tuned_model $save_path \

    iteration=$((iteration + 3))

done

