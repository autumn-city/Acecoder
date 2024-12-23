import argparse
import os
import json
from tqdm import tqdm
import copy
import openai
import sys
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from datasets import load_dataset
import time
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from torch.utils.data import DataLoader
import pickle
from dataset import build_dataset_in_dataframe


def collator(data):
    batch = dict((key, [d[key] for d in data]) for key in data[0])
    return batch

def fetch_completion(data_entry_lists, llm, sampling_params):
    input_batch = []
    for entry in data_entry_lists:
        input_batch.append(entry['question_description'])
    generation = llm.generate(input_batch, sampling_params)
    generated_text = []
    for i in range(len(generation)):
        # generated_text.append(generation[i].outputs[0].text)
        data_entry_lists[i]['generated_code'] = generation[i].outputs[0].text
    # data_entry_lists['generated_code'] = generation
    return data_entry_lists


if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument(
        "--checkpoint",
        type=str,
        default="m-a-p/OpenCodeInterpreter-DS-1.3B",
        required=True,
    )    
    
    args.add_argument(
        "--batch_size",
        type=int,
        default=8,
        required=True,
    )        
    
    args.add_argument(
        "--number_0f_GPUs",
        type=int,
        default=1,
        required=True,
    )
    
        
    args.add_argument(
        "--gpu_usage_per",
        type=float,
        default=1.0,
        required=True,
    )
    
    args.add_argument(
        "--output",
        type=str,
        default='./results',
        required=True,
    )    
    
    args = args.parse_args()
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    gpu_number = args.number_0f_GPUs
    gpu_usage_per = args.gpu_usage_per
    output_dir = args.output

    with open("evaluation_dataset/test.json", "r") as f:
        test_data = json.load(f)

    llm = LLM(
        model=checkpoint,
        tensor_parallel_size=gpu_number,
        gpu_memory_utilization=gpu_usage_per,
        max_model_len=4096,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0, 
        top_p=1, 
        top_k=-1, 
        max_tokens=1024, 
        stop=[tokenizer.eos_token], 
    )

    for i in tqdm(range(0, len(test_data), batch_size)):
    # for i in tqdm(range(0, 16, batch_size)):
        # Process each batch
        batch_end = min(i + batch_size, len(test_data))
        
        test_data[i:batch_end] = fetch_completion(test_data[i:batch_end], llm, sampling_params)
        
    print(test_data)
        
    record_name = checkpoint.split("/")[-1]        
    
    print(f"Saving the results to {output_dir}/{record_name}.json")
    
    with open(f'{output_dir}/{record_name}.json', 'w') as f:
        json.dump(test_data, f)
    
