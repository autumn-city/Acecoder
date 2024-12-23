import sys
sys.path.append('/workspace')
import argparse
import re
import json
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from timing import code_executing, execute_multiple_codes_in_order
from execution_reward import reward_calculator, calculate_runtime
import multiprocessing
import signal
import os
import pandas
from timing import lib_complete    
from code_execute_v2 import code_excutor

logger = init_logger(__name__)

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text


class RewardModelProxy:
    def __init__(self, args):

        self.questions = json.load(open('/workspace/EffiBench/data/dataset.json'))
        self.power = args.power
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    
    def extract_task_description(self, text):
        # Regular expression to extract the task description
        pattern = r"# Task description:\n```python\n(.*?)\n```"
        task_description = re.search(pattern, text, re.DOTALL)
 
        # Extract the task description if found
        extracted_task_description = task_description.group(1) if task_description else "No task description found."
        return extracted_task_description

    def question_id_identify(self, input_list):
        r = []
        for text in input_list:
            extracted_task_description = self.extract_task_description(text)
            # print with the special token like \n
            # print(extracted_task_description)
            for question in self.questions:
                if clean_string(extracted_task_description) in clean_string(question['markdown_description']):
                    r.append(question['problem_idx'])
                    break
        return r
    
    def code_extract(self, prompts):
        codes = []
        for prompt in prompts:
            try:
                code = re.findall(r'```python(.*?)```', prompt, re.DOTALL)[-1]
            except:
                code = "No code found."
            codes.append(code)
        return codes
    def reward_4_code_execute(self, code, id):
        print('start the code execution')

        executed_result = update_code_executing(code, id)
        # print(executed_result)
        logger.info(f"Execution Result: {executed_result}")
        gt_execution_result = self.gt_code_execute(id)
        
        scores = {}
        

        for idx in executed_result.keys():
            q_id = id[int(idx)]
            ref_result = gt_execution_result[str(idx)]
            gen_result = executed_result[str(idx)]
            stat = gen_result[0]
            text_result = gen_result[1]
            if stat == 0:
                efficiency = ref_result[1] / text_result
                scores[idx] = 0.5+0.5*(min(1,efficiency**self.power))
            elif stat == 1 and 'TimeoutError' not in text_result:
                scores[idx] = -0.3
            elif stat == 1 and 'TimeoutError' in text_result:
                scores[idx] = -0.2
        
        # return the values in order of the input
        return [scores[str(i)] for i in range(len(code))]
        
        
    
    def gt_code_execute(self, id):
        gt_code = {}
        gt_result = {}
        source_path = '/workspace/data/generated_bad_code/reference_human_solution/'
        for idx, q_id in enumerate(id):
            q_file = source_path + str(q_id) + '.py'
            with open(q_file, 'r') as file:
                code = file.read()
                gt_code[str(idx)] = code
                
        gt_result = code_excutor(gt_code)
        logger.info(f"GT Execution Result: {gt_result}")
        # print('the gt code is :', gt_code)
        # print('the gt id is :', id)
        return gt_result

dataset4testcase = pandas.read_json('../EffiBench/data/dataset.json')
global test_cases 
test_cases = {}
for idx, test_case in zip(dataset4testcase['problem_idx'], dataset4testcase['test_case']):
    test_cases[idx] = test_case

def update_code_executing(generated_code, ids):
    complete_codes = {}
    for idx, code in enumerate(generated_code):
        # concat the test cases and generated code 
        q_id = ids[idx]
        test_case = test_cases[q_id]
        if isinstance(code, list):
            complete_code = "\nsolution=Solution()\n" + test_case
        if isinstance(code, str):
            complete_code = code+"\nsolution=Solution()\n" + test_case
        complete_code = lib_complete(complete_code)
        complete_codes[str(idx)] = complete_code
    
    logger.info('the len of complete codes:%d', len(complete_codes))
            
    results =  code_excutor(complete_codes)
    return results


    
    
def clean_string(s):
    # Use regex to remove all whitespace characters and special tokens
    return re.sub(r'\s+|\n+', '', s)

def test():
    
    QuestionID = [1009, 2656, 1012, 2656]
    Codes= ['\nclass Solution:\n    def bitwiseComplement(self, n: int) -> int:\n        bits = list(bin(n))[2:]\n        for i, value in enumerate(bits):\n            bits[i] = "1" if value == "0" else "0"\n        \n        return int("".join(bits), 2)\n', '\nfrom typing import *\n\nclass Solution:\n    def maximizeSum(self, nums: List[int], k: int) -> int:\n        nums.sort(reverse=True)\n        score = 0\n        for i in range(k):\n            score += nums[i]\n            nums.append(nums[i] + 1)\n            nums.sort(reverse=True)\n        return score\n', '\nclass Solution:\n    def numDupDigitsAtMostN(self, n: int) -> int:\n        def count(n, dp):\n            if dp[n] is not None:\n                return dp[n]\n            if n == 0 or n == 1:\n                return 0\n            elif n == 2:\n                dp[n] = 1\n            else:\n                dp[n] = count(n-1, dp) * (10 - n + 2)\n                if n > 10:\n                    dp[n] -= count(n-2, dp) * 2\n            return dp[n]\n\n        s = str(n)\n        dp = [None] * (len(s) + 1)\n        dp[0] = 0\n\n        ans = 0\n        for i in range(1, len(s)):\n            ans += 9 * 9 * (10**(i-1))\n        ans += count(len(s), dp)\n\n        visited = [False] * 10\n        for i in range(len(s)):\n            if i != 0:\n                visited[int(s[i-1])] = True\n            for j in range(1 if i == 0 else 0, int(s[i])):\n                if not visited[j]:\n                    ans += 10**(len(s) - i - 1)\n            if visited[int(s[i])]:\n                break\n        return n - ans\n']    
    Codes.append('class Solution:\n    import time\n    time.sleep(0.1)')
    logger.info('len of codes:%d', len(Codes))
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")
    parser.add_argument("--max_len", type=int, default="2048")
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument('--power', type=int, default=1, help='Power of the reward')

    args = parser.parse_args()

    # server
    reward_model = RewardModelProxy(args)

    print(reward_model.reward_4_code_execute(Codes, QuestionID))



def signal_handler(sig, frame):
    """Handles shutdown signals to ensure all processes are terminated properly."""
    print('Shutdown signal received. Cleaning up...')
    # This will terminate all child processes before shutting down
    multiprocessing.active_children()
    os._exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")
    parser.add_argument("--max_len", type=int, default="2048")
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument('--power', type=int, default=1, help='Power of the reward')

    args = parser.parse_args()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()
    
    signal.signal(signal.SIGINT, signal_handler)

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        # logger.info(f"Received JSON: {data}")
        queries = data.get("query")
        # logger.info(f"Queries: {queries}")
        # rewards = reward_model.get_reward(queries)
        id = reward_model.question_id_identify(queries)
        logger.info(f"Question ID: {id}")
        codes = reward_model.code_extract(queries)
        # logger.info(f"Codes: {codes}")    
        execution_reward = reward_model.reward_4_code_execute(codes, id)
        logger.info(f"Execution Reward: {execution_reward}")  
        logger.info(f"Reward: {execution_reward}")    
        # return JSONResponse(list(execution_reward.values()))
        result = {"rewards": execution_reward}
        # result = {"rewards": rewards}
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    
    # test()