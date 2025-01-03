# Replication Package for "Enhancing Efficiency in LLM-Generated Code"

This repository contains the replication package for the paper "ACECode: A Reinforcement Learning Framework for Aligning Code Efficiency and Correctness in Code Language Models". The package includes all materials needed to reproduce the results presented in the paper.

# Dataset

The dataset used in this replication package consists of 8520 code solutions to 797 LeetCode problems. These solutions have been crawled and collected from LeetCode forum for evaluating the efficiency and correctness of code generated by LLMs. The dataset includes human-written solutions written in Python3. The test cases are derived from EffiBench.

The dataset can be accessed and downloaded from the following link:
[Download Dataset](https://smu-my.sharepoint.com/:u:/g/personal/cryang_smu_edu_sg/EYpufI7GUQNPnhdm-I1tvWwBk5_w1KAM-N-cb25VuTKs7Q?e=ukjpFv)


# Training Pipeline

## Pre-requirement:

```
Setting up the OpenRLHF environment following the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
```

## Setting Rewarding Environment before training

```
remote_reward.sh
```

## Training

```
efficode_train_multi_epoch.sh
```

# Evaluation

Execute the following script to generate the code from LLM. Note that we apply vllm to evaluate the performance of the model. Please refer to the [vllm](https://github.com/vllm-project/vllm) for vllm setting.
```
python vllm_evaluation.py \
    --checkpoint $ref_model \ # the path of the model to be evaluated
    --batch_size 4 \
    --number_0f_GPUs 1 \
    --gpu_usage_per 0.5 \
    --output ./baseline_result

```
execute the following script to compare the execution time for each generated code between the baseline and ACECode.
```
execution_time_statistic_async.sh
```
 
