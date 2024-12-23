import argparse
import os
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
import timeit
import re
from code_execute import lib_complete
import traceback
import signal
from functools import wraps
import logging

args = argparse.ArgumentParser()
args.add_argument(
    "--ref_model",
    type=str,
    default="m-a-p/OpenCodeInterpreter-DS-1.3B",
    required=True,
)
args.add_argument(
    "--tuned_model",
    type=str,
    default=1,
    required=True,
)
args.add_argument(
    "--functimeout",
    type=int,
    default=1,
    required=True,
)

args.add_argument(
    "--accumulate_time",
    type=float,
    default=1.5,
    required=True,        
)


args = args.parse_args()
ref_model = args.ref_model
tuned_model = args.tuned_model
func_timeout = args.functimeout
accumulate_time = args.accumulate_time

def logger_setting(model_name_):
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_path='logs/'+model_name_+'_timeout_'+str(func_timeout)+'_accumulate_'+str(accumulate_time)+'.log'
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    import warnings
    warnings.filterwarnings("ignore", message=".*parallelism.*", category=UserWarning)
    
    return logger


def process_key(key, ref_response, target_response, test_cases, results_in_common, results_all):
    line_record = {}
    test_case = test_cases[int(key)]
    target_code = code_extract(target_response[key], test_case)
    ref_code = code_extract(ref_response[key], test_case) 
    
    ref_flag, ref_time = execution_time_statistic(ref_code)
    line_record['ref_flag'] = ref_flag
    line_record['ref_time'] = ref_time
    line_record['key'] = key
    print('ref_flag: ', ref_flag)          
    
    target_flag, target_time = execution_time_statistic(target_code)
    line_record['target_flag'] = target_flag
    line_record['target_time'] = target_time
    print('target_flag: ', target_flag)     
    
    if ref_flag == 0 and target_flag == 0:
        results_in_common[key] = (ref_time, target_time)
    
    print(line_record)
    return line_record

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

class CustomTimer(timeit.Timer):
    
    # Step 2: Override the autorange method
    def autorange(self, callback=None):
        """Return the number of loops and time taken so that total time >= 1.5.
        
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
                if time_taken >= accumulate_time:
                    return (number, time_taken)
            i *= 10

def get_data(model):
    record_name = model.split("/")[-1]      
    with open(f"results/{record_name}.json", "r") as f:
        data = json.load(f)
    return data

def get_response(data):
    # from json to dict with 'id' as key and 'generated_code' as value
    response = {}
    for item in data:
        response[item['id']] = item['generated_code']
    return response

def code_extract(prompt, test_case):
    try:
        code = re.findall(r'```python(.*?)```', prompt, re.DOTALL)[0]
        complete_code = code+"\nsolution=Solution()\n" + test_case
        complete_code = lib_complete(complete_code)
        return complete_code
    except:
        return "\nsolution=Solution()\n" + test_case

@timeout(func_timeout)
def execution_time_statistic(code):

    setup = code.split('class Solution:')[0]
    code_content = code.split('class Solution:')[1]
    code_content = 'class Solution:\n' + code_content
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
        
    
def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--ref_model",
        type=str,
        default="m-a-p/OpenCodeInterpreter-DS-1.3B",
        required=True,
    )
    args.add_argument(
        "--tuned_model",
        type=str,
        default=1,
        required=True,
    )
    args = args.parse_args()
    ref_model = args.ref_model
    tuned_model = args.tuned_model
    
    ref_data = get_data(ref_model)
    target_data = get_data(tuned_model)
    
    ref_response = get_response(ref_data)
    target_response = get_response(target_data)
    
    assert len(ref_response) == len(target_response)
    
    # init the test cases
    import pandas
    dataset4testcase = pandas.read_json('../../../EffiBench/data/dataset.json')    
    global test_cases 
    test_cases = {}
    for idx, test_case in zip(dataset4testcase['problem_idx'], dataset4testcase['test_case']):
        test_cases[idx] = test_case    
    
    results = {}
    
    # with ThreadPoolExecutor() as executor:
    for key in ref_response.keys():
        test_case = test_cases[int(key)]
        # print('ref_response[key]: ', ref_response[key])            
        ref_code = code_extract(ref_response[key], test_case)
        target_code = code_extract(target_response[key], test_case)
        ref_flag, ref_time = execution_time_statistic(ref_code)
        target_flag, target_time = execution_time_statistic(target_code)              
        if ref_flag == 0 and target_flag == 0:
            results[key] = (ref_time, target_time)
    # save the results
    record_name = tuned_model.split("/")[-1]       
    with open(f'results/{record_name}_execution_time.json', 'w') as f:
        json.dump(results, f)

def statistic():
    for i in range(3, 33, 3):
        meta_data = f'results/magicoder-6.7B-epoch-{i}_execution_time.json'
        with open(meta_data, 'r') as f:
            data = json.load(f)
        # if len([1 for key in data.keys() if data[key][1] < data[key][0]]) > len(data)/2:
        print(meta_data)
        print('the length of data: ', len(data))
        print('%d target time is less than ref time' % len([1 for key in data.keys() if data[key][1] < data[key][0]]))
        print('%d target time is less than ref time' % len([1 for key in data.keys() if data[key][1] < 0.95*data[key][0]]))
        print('%d target time is larger than ref time' % len([1 for key in data.keys() if data[key][1]*0.95 > data[key][0]]))
        # count the all the target time and ref time
        ref_time = 0
        target_time = 0
        for key in data.keys():
            ref_time += data[key][0]
            target_time += data[key][1]
        print('the total ref time: ', ref_time)
        print('the total target time: ', target_time)
        print('\n'*3)
            
def calculate_pass1():
        
    # init the test cases
    import pandas
    dataset4testcase = pandas.read_json('../../../EffiBench/data/dataset.json')    
    global test_cases 
    test_cases = {}
    for idx, test_case in zip(dataset4testcase['problem_idx'], dataset4testcase['test_case']):
        test_cases[idx] = test_case    
    
    results_in_common = {}
    
    results_all = []
    
    # Maximum allowed execution time for each task
    MAX_EXECUTION_TIME = 10

    num_cores = os.cpu_count()
    max_workers = num_cores//2
    logger.info('number of  workers: %s', max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_key, key, ref_response, target_response, test_cases, results_in_common, results_all): key for key in ref_response.keys()}
        
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result(timeout=MAX_EXECUTION_TIME)
                results_all.append(result)
            except TimeoutError:
                print(f"Task for key {key} exceeded the max execution time of {MAX_EXECUTION_TIME} seconds and was terminated.")
            except Exception as e:
                print(f"Task for key {key} failed with an exception: {e}")
                
    # save the results in common
    record_name = tuned_model.split("/")[-1]       
    with open(f'results/{record_name}_execution_time.json', 'w') as f:
        json.dump(results_in_common, f)
    # save the results all in jsonl
    import pandas as pd
    df = pd.DataFrame(results_all)
    target_file = f"results/{record_name}_execution_all_record.jsonl"
    df.to_json(target_file, orient='records', lines=True)
    # evaluation_metric(f'results/{record_name}_execution_all_record.jsonl')
    metric_report(logger, target_file)
    
def evaluation_metric(meta_data):
    
    # meta_data = 'results/magicoder-6.7B-epoch-12_execution_all_record.jsonl'
    import pandas as pd
    df = pd.read_json(meta_data, orient='records', lines=True)
        
    logger.info('the pass@1 of ref model: %s', len(df[df['ref_flag'] == 0])/len(df))
    logger.info('the pass@1 of target model: %s', len(df[df['target_flag'] == 0])/len(df))
    
    # for those that ref_flag and target_flag are both 0, count the total ref_time and target_time
    ref_time = 0
    target_time = 0
    ref_time_multi = 0
    target_time_multi = 0
    achievement = 0
    in_common = 0
    for index, row in df.iterrows():
        if row['ref_flag'] == 0 and row['target_flag'] == 0:
            in_common+=1
            ref_time += row['ref_time']
            target_time += row['target_time']
            if ref_time_multi == 0:
                ref_time_multi = row['ref_time']
            else:
                ref_time_multi *= row['ref_time']
            if target_time_multi == 0:
                target_time_multi = row['target_time']
            else:
                target_time_multi *= row['target_time']
            if row['target_time'] < row['ref_time']:
                achievement += 1
            
    
    # logger.info('the total ref time: %s', ref_time)
    # logger.info('the total target time: %s', target_time)
    # logger.info('Percentage of average execution time reduced from target model comparing to ref model: %s', (ref_time - target_time)/ref_time)
    # logger.info('Percentage of functions which execution time reduces from target model comparing to ref model: %s', achievement/in_common)
    # logger.info('geometric mean of ref time: %s', ref_time_multi**(1/in_common))
    # logger.info('geometric mean of target time: %s', target_time_multi**(1/in_common))
    
    logger.info('AET of target: %s', (ref_time)/in_common)
    logger.info('AET of ref: %s', (target_time)/in_common)
    logger.info('AET improvement: %s', ((ref_time)/in_common-(target_time)/in_common)/((ref_time)/in_common))
    logger.info('NAET: %s', (ref_time)/target_time)
    logger.info('ECC: %s', achievement/in_common)
    logger.info('GET: %s', target_time_multi**(1/in_common))
    logger.info('Reference GET: %s', ref_time_multi**(1/in_common))
    logger.info('relative GET (a negative percentage means efficiency improvement) %s', ((target_time_multi**(1/in_common))-(ref_time_multi**(1/in_common)))/(ref_time_multi**(1/in_common)))
    logger.info('overlapping rate: %s', in_common/len(df))

def metric_report(logger,target_file):

    logger.info('\n'*2)
    logger.info('======')
    logger.info('checkpoint: %s', target_file)
    evaluation_metric(target_file)
        

        
if __name__ == "__main__":
    
    ref_data = get_data(ref_model)
    target_data = get_data(tuned_model)
    
    ref_response = get_response(ref_data)
    
    print(ref_response);exit()
    
    target_response = get_response(target_data)
    
    assert len(ref_response) == len(target_response)
    
    model_name_ = tuned_model.split("/")[-1].split('-')[0]
    
    logger = logger_setting(model_name_)
    
    calculate_pass1()
    
    