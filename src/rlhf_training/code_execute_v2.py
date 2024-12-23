import argparse
import os
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
import timeit
import re
import traceback
import signal
from functools import wraps
import logging
from memory_efficiency_evaluator.capture_tracemalloc_value import run_and_capture_peak_value

# Define the timeout decorator
def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Define the handler for the timeout signal
            def handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set the signal handler and a timeout alarm
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm after the function completes
                signal.alarm(0)

            return result
        return wrapper
    return decorator

@timeout(10)
def execution_time_statistic(id, code):

    setup = code.split('solution=Solution()')[0]
    code_content = code.split('solution=Solution()')[1]
    code_content = 'solution=Solution()' + code_content
    try:
        iteration, execution_times = CustomTimer(
        code_content.strip(),
        setup=setup.strip(),
        globals=globals()
    ).autorange()
        return (0, execution_times / iteration)
    except Exception :
        stacktrace = traceback.format_exc()
        return (1, stacktrace)
    
@timeout(10)
def execution_memory_statistic(id, code):
    
    print(code+'\n')
    
    peack_memory = run_and_capture_peak_value(code, 10)
    
    if peack_memory is not None:
        return (0, peack_memory)
    else:
        return (1, 'No peak memory value found')
    
    
def code_excutor(code_dict):
    results_all = {}            
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(execution_time_statistic, key, code_dict[key]): key for key in code_dict.keys()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                results_all[key] = result
            except TimeoutError:
                results_all[key] = (1, 'Timeout')
            
    return results_all

def code_excutor_memory_profile(code_dict):
    results_all = {}            
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(execution_memory_statistic, key, code_dict[key]): key for key in code_dict.keys()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                results_all[key] = result
            except TimeoutError:
                results_all[key] = (1, 'Timeout')
            
    return results_all

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
                if time_taken >= 1:
                    return (number, time_taken)
            i *= 10