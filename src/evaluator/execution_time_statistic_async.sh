model_path="checkpoints/CodeQwen/CodeQwen1.5-7B-Chat"
ref_model="Qwen/CodeQwen1.5-7B-Chat"
start_iteration=3
max_iteration=30 # You can modify this to your desired maximum
iteration=$start_iteration


while [ $iteration -le $max_iteration ]; do
    save_path="${model_path}-${iteration}"
    python execution_time_statistic_async.py \
        --ref_model $ref_model \
        --tuned_model $save_path 

    iteration=$((iteration + 3))

done