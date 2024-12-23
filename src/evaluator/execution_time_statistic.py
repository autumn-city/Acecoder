import argparse
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import timeit
import re
from code_execute import lib_complete
import traceback
import signal
from functools import wraps

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
                if time_taken >= 1.5:
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

@timeout(10)
def execution_time_statistic(code):

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
        print('ref_flag: ', ref_flag)
        target_flag, target_time = execution_time_statistic(target_code)
        print('target_flag: ', target_flag)
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
    
    results_in_common = {}
    
    results_all = []
    
    # with ThreadPoolExecutor() as executor:
    for key in ref_response.keys():
        line_record = {}
        test_case = test_cases[int(key)]
        ref_code = code_extract(ref_response[key], test_case)
        target_code = code_extract(target_response[key], test_case)
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
        results_all.append(line_record)

    # save the results in common
    record_name = tuned_model.split("/")[-1]       
    with open(f'results/{record_name}_execution_time.json', 'w') as f:
        json.dump(results_in_common, f)
    # save the results all in jsonl
    import pandas as pd
    df = pd.DataFrame(results_all)
    print('df: ', df)
    df.to_json(f'results/{record_name}_execution_all_record.jsonl', orient='records', lines=True)
    evaluation_metric(f'results/{record_name}_execution_all_record.jsonl')
    
def evaluation_metric(meta_data):
    
    # meta_data = 'results/magicoder-6.7B-epoch-12_execution_all_record.jsonl'
    import pandas as pd
    df = pd.read_json(meta_data, orient='records', lines=True)
    # count the number of ref_flag equal to 1
    print('the number of ref_flag equal to 0: ', len(df[df['ref_flag'] == 0]))
    print('the pass@1 of ref model: ', len(df[df['ref_flag'] == 0])/len(df))
    print('the number of target_flag equal to 0: ', len(df[df['target_flag'] == 0]))
    print('the pass@1 of target model: ', len(df[df['target_flag'] == 0])/len(df))
    
    # count the number that ref_flag and target_flag are both 0
    print('the number that ref_flag and target_flag are both 0: ', len(df[(df['ref_flag'] == 0) & (df['target_flag'] == 0)]))
    # for those that ref_flag and target_flag are both 0, count the total ref_time and target_time
    ref_time = 0
    target_time = 0
    achievement = 0
    for index, row in df.iterrows():
        if row['ref_flag'] == 0 and row['target_flag'] == 0:
            ref_time += row['ref_time']
            target_time += row['target_time']
            if row['target_time'] < row['ref_time']:
                achievement += 1
    print('the total ref time: ', ref_time)
    print('the total target time: ', target_time)
    print('the achievement of target model: ', achievement)
    print('AET: ', target_time/len(df))
    print('NAET: ', ref_time/target_time)
    # geometric mean of target_time
    # print('GET:', target_time**(1/len(df)))
    # print('NGET:', (target_time/ref_time)**(1/(len(df))))


if __name__ == "__main__":
    # main()
    # statistics = statistic()
    calculate_pass1()
    # evaluation_metric('results/magicoder-6.7B-epoch-24_execution_all_record.jsonl')
    
    
    # set epoch 9 and peoch 12 for inference
    # define the inference metric