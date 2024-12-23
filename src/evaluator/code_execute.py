import subprocess
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import pandas

dataset4testcase = pandas.read_json('../../../EffiBench/data/dataset.json')
global test_cases 
test_cases = {}
for idx, test_case in zip(dataset4testcase['problem_idx'], dataset4testcase['test_case']):
    test_cases[idx] = test_case

def insert_at_beginning(original_string, string_to_insert):
    return string_to_insert + original_string      

def lib_complete(code):
    
    if 'List' in code:
        code = insert_at_beginning(code,'from typing import List\n')
    if 'cache' in code:
        code = insert_at_beginning(code,'from functools import cache\n')
    if 'Optional' in code:
        code = insert_at_beginning(code,'from typing import Optional\n')
    if 'root' in code:
        code = insert_at_beginning(code,'from math import sqrt\n')
    if 'ListNode' in code:
        code = insert_at_beginning(code,'class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n')
    if 'gcd' in code:
        code = insert_at_beginning(code,'from math import gcd\n')
    if 'deque' in code:
        code = insert_at_beginning(code,'from collections import deque\n')
    if 'defaultdict' in code:
        code = insert_at_beginning(code,'from collections import defaultdict\n')
    if 'bisect_left' in code:
        code = insert_at_beginning(code,'from bisect import bisect_left\n')
    if 'lcm' in code:
        code = insert_at_beginning(code,'from math import lcm\n')
    if 'aq' in code:
        code = insert_at_beginning(code,'from collections import deque\n')
    if 'ascii_lowercase' in code:
        code = insert_at_beginning(code,'from string import ascii_lowercase\n')
    if 'TreeNode' in code:
        code = insert_at_beginning(code,'''
class TreeNode:
def __init__(self, val=0, left=None, right=None):
self.val = val
self.left = left
self.right = right\n\n''')            
    if 'inf' in code:
        code = insert_at_beginning(code,'from math import inf\n')
    if 'Counter' in code:
        code = insert_at_beginning(code,'from collections import Counter\n')
    if 'cmp_to_key' in code:
        code = insert_at_beginning(code,'from functools import cmp_to_key\n')
    if 'pairwise' in code:
        code = insert_at_beginning(code,'from itertools import pairwise\n')
    if 'reduce' in code:
        code = insert_at_beginning(code,'from functools import reduce\n')
    if 'heappush' in code:
        code = insert_at_beginning(code,'from heapq import heappush\n')
    if 'heapify' in code:
        code = insert_at_beginning(code,'from heapq import heapify\n')
    if 'heappop' in code:
        code = insert_at_beginning(code,'from heapq import heappop\n')
    if 'nlargest' in code:
        code = insert_at_beginning(code,'from heapq import nlargest\n')
    if 'math' in code:
        code = insert_at_beginning(code,'import math\n')
    if 'accumulate' in code:
        code = insert_at_beginning(code,'from itertools import accumulate\n')
    if 'randint' in code:
        code = insert_at_beginning(code,'from random import randint\n')
    if 'product' in code:
        code = insert_at_beginning(code,'from itertools import product\n')
    if 'nlargest' in code:
        code = insert_at_beginning(code,'from heapq import nlargest\n')
    if 'bisect_right' in code:
        code = insert_at_beginning(code,'from bisect import bisect_right\n')
    if 'permutations' in code:
        code = insert_at_beginning(code,'from itertools import permutations\n')
    if 'ceil' in code:
        code = insert_at_beginning(code,'from math import ceil\n')
    if 'sys' in code:
        code = insert_at_beginning(code,'import sys\n')
    if 'loads' in code:
        code = insert_at_beginning(code,'from json import loads\n')
    if 'heapq' in code:
        code = insert_at_beginning(code,'import heapq\n')
    if 'lru_cache' in code:
        code = insert_at_beginning(code,'from functools import lru_cache\n')
    if 'Set' in code:
        code = insert_at_beginning(code,'from typing import Set\n')
    if 'combinations' in code:
        code = insert_at_beginning(code,'from itertools import combinations\n')
    if 'collections' in code:
        code = insert_at_beginning(code,'import collections\n')
    if 'stdin' in code:
        code = insert_at_beginning(code,'from sys import stdin\n')
    if 'heappushpop' in code:
        code = insert_at_beginning(code,'from heapq import heappushpop\n')
    if 'comb' in code:
        code = insert_at_beginning(code,'from math import comb\n')
    if 'factorial' in code:
        code = insert_at_beginning(code,'from math import factorial\n')
    if 'xor' in code:
        code = insert_at_beginning(code,'from operator import xor\n')
    if 'bisect' in code:
        code = insert_at_beginning(code,'from bisect import bisect\n')
    if 'log' in code:
        code = insert_at_beginning(code,'from math import log\n')
    if 'inf' in code:
        code = insert_at_beginning(code,'from math import inf\n')
    if 'itemgetter' in code:
        code = insert_at_beginning(code,'from operator import itemgetter\n')
    if 'copy' in code:
        code = insert_at_beginning(code,'from copy import copy\n')
    if 'itertools' in code:
        code = insert_at_beginning(code,'import itertools\n')
    if 'random' in code:
        code = insert_at_beginning(code,'import random\n')
    if 'nsmallest' in code:
        code = insert_at_beginning(code,'from heapq import nsmallest\n')
    if 'log2' in code:
        code = insert_at_beginning(code,'from math import log2\n')
    if 'floor' in code:
        code = insert_at_beginning(code,'from math import floor\n')
    if 'Union' in code:
        code = insert_at_beginning(code,'from typing import Union\n')
    if 'wordSet' in code:
        code = insert_at_beginning(code,'from typing import Set\n')
    if 'Tuple' in code:
        code = insert_at_beginning(code,'from typing import Tuple\n')
    if 'DirectoryTrieNode' in code:
        code = insert_at_beginning(code,'class DirectoryTrieNode:\n    def __init__(self):\n        self.children = {}\n        self.isEnd = False\n')
    if 'perm' in code:
        code = insert_at_beginning(code,'from itertools import permutations\n')
    if 'PQ' in code:
        code = insert_at_beginning(code,'from queue import PriorityQueue\n')
    if 'ascii_uppercase' in code:
        code = insert_at_beginning(code,'from string import ascii_uppercase\n')
    if 'Iterator' in code:
        code = insert_at_beginning(code,'from typing import Iterator\n')
    if 'functools' in code:
        code = insert_at_beginning(code,'import functools\n')
    if 'repeat' in code:
        code = insert_at_beginning(code,'from itertools import repeat\n')
    if 'comb' in code:
        code = insert_at_beginning(code,'from math import comb\n')
    if 'bisect' in code:
        code = insert_at_beginning(code,'from bisect import bisect\n')
    if 'DefaultDict' in code:
        code = insert_at_beginning(code,'from collections import defaultdict\n')
    if 'operator' in code:
        code = insert_at_beginning(code,'import operator\n')
    if 'string' in code:
        code = insert_at_beginning(code,'import string\n')
    if 'groupby' in code:
        code = insert_at_beginning(code,'from itertools import groupby\n')
    if 'Dict' in code:
        code = insert_at_beginning(code,'from typing import Dict\n')
    if 'Row' in code:
        code = insert_at_beginning(code,'from typing import List\n')
    if 'log2' in code:
        code = insert_at_beginning(code,'from math import log2\n')
    if 'deepcopy' in code:
        code = insert_at_beginning(code,'from copy import deepcopy\n')
    if 'sqrt' in code:
        code = insert_at_beginning(code,'from math import sqrt\n')
    if 'maxsize' in code:
        code = insert_at_beginning(code,'from sys import maxsize\n')    
    return code

def code_executing(generated_code, ids):
    complete_codes = []
    for idx in generated_code.keys():
        # concat the test cases and generated code 
        code = generated_code[idx]
        
        test_case = test_cases[int(idx)]
        if isinstance(code, list):
            complete_code = "\nsolution=Solution()\n" + test_case
        if isinstance(code, str):
            complete_code = code+"\nsolution=Solution()\n" + test_case
        complete_code = lib_complete(complete_code)
        complete_codes.append(complete_code)
        # print('complete_code', complete_code)
        # print('idx', idx)
    results = execute_multiple_codes_in_order(complete_codes)
    return results

from concurrent.futures import ProcessPoolExecutor
import asyncio
import traceback
import timeit
import multiprocessing
executor = ProcessPoolExecutor()

class CustomTimer(timeit.Timer):
    
    # Step 2: Override the autorange method
    def autorange(self, callback=None):
        """Return the number of loops and time taken so that total time >= 0.5.
        
        This is the customized version of the autorange method.
        """
        i = 1
        while True:
            for j in 1, 2, 5:
                number = i * j
                time_taken = self.timeit(number)
                if callback:
                    callback(number, time_taken)
                # Step 3: Modify the condition
                if time_taken >= 0.5:
                    return (number, time_taken)
            i *= 10

def execute_and_time_code(code, timeout=10):
    try:
        # Extract setup and code_content

        setup = code.split('class Solution:')[0]
        code_content = code.split('class Solution:')[1]
        code_content = 'class Solution:\n' + code_content

        execute_per_times = []

        for i in range(5):
            # Define the function to be run in a separate process
            def target(return_dict):
                try:
                    # Here, we can set up the timer
                    iteration, execution_times = CustomTimer(
                        code_content.strip(),
                        setup=setup.strip(),
                        globals=globals()
                    ).autorange()
                    return_dict['result'] = execution_times / iteration
                except Exception:
                    stacktrace = traceback.format_exc()
                    return_dict['error'] = stacktrace

            # Now run the target function in a separate process
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = multiprocessing.Process(target=target, args=(return_dict,))
            p.start()
            p.join(timeout)
            if p.is_alive():
                kill_proc_tree(p.pid)
                p.join()
                return (0, "Timeout")
            else:
                if 'error' in return_dict:
                    return (1, return_dict['error'])
                else:
                    execute_per_times.append(return_dict['result'])
        return (0, min(execute_per_times))
    except Exception:
        # If there's an error during compilation, return the stacktrace (now detect whether the class is correctly generated)
        stacktrace = traceback.format_exc()
        return (2, stacktrace)



# def execute_multiple_codes_in_order(codes_list, max_execution_time=10):
#     loop = asyncio.get_running_loop()
#     results = [None] * len(codes_list)

#     # Use a ProcessPoolExecutor
#     # with ProcessPoolExecutor() as executor:
#         # Schedule the execution of each code snippet
#     tasks = [
#         loop.run_in_executor(
#             executor,
#             execute_and_time_code,
#             code
#         )
#         for code in codes_list
#     ]

#     # Iterate over the tasks and enforce timeout on each one
#     for idx, task in enumerate(tasks):
#         try:
#             # Use asyncio.wait_for to enforce the max_execution_time on each task
#             result = asyncio.wait_for(task, timeout=max_execution_time)
#             results[idx] = result
#             # print(f"Task {idx} completed with result: {result}")
#         except asyncio.TimeoutError:
#             # print(f"Task {idx} timed out, cancelling...")
#             results[idx] = (0, "Timeout")
#             task.cancel()  # Cancel the task if it times out
#         except Exception as e:
#             # print(f"Task {idx} completed with error: {e}")
#             results[idx] = (2, f"Error: {e}")

#     return results

import concurrent.futures


def execute_multiple_codes_in_order(codes_list, max_execution_time=10):
    results = [None] * len(codes_list)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(execute_and_time_code, code): idx for idx, code in enumerate(codes_list)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result(timeout=max_execution_time)
                results[idx] = result
                # print(f"Task {idx} completed with result: {result}")
            except concurrent.futures.TimeoutError:
                # print(f"Task {idx} timed out, cancelling...")
                results[idx] = (0, "Timeout")
                future.cancel()
            except Exception as e:
                # print(f"Task {idx} completed with error: {e}")
                results[idx] = (2, f"Error: {e}")
    return results

import psutil

def kill_proc_tree(pid, including_parent=True, timeout=5):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    gone, still_alive = psutil.wait_procs(children, timeout=timeout)
    if including_parent:
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        parent.wait(timeout)