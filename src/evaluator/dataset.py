import torch
from tqdm import tqdm
import pandas as pd
import os
tqdm.pandas()
import re
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
import json
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline, AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Access the row by index
        row = self.dataframe.iloc[idx]
        
        # Extract columns
        code = str(row['code'])
        question_description = str(row['question_description'])
        score = row['score']
        question_id = row['id']
        
        # Tokenize the inputs
        query = torch.tensor(self.tokenizer.encode(question_description))
        response = torch.tensor(self.tokenizer.encode(code))
        
        # Return a dictionary with all desired columns
        return {
            'query': query,
            'response': response,
            'score': torch.tensor(score, dtype=torch.float32),
            'id': question_id,
            'code': code,
            'question_description': question_description
        }
        
def remove_comments_and_blank_lines(code):
    # Remove single-line comments (start with #)
    code_no_single_comments = re.sub(r'#.*', '', code)
    
    # Remove multi-line comments (enclosed in '''...''' or """...""")
    code_no_comments = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code_no_single_comments)
    
    # Remove blank lines (lines that are just whitespace or completely empty)
    code_no_blank_lines = "\n".join([line for line in code_no_comments.splitlines() if line.strip()])
    
    return code_no_blank_lines
        
        
def build_datasetg_leetcode(model_id, test_list):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # load training dataset
    import pickle

    code_store = '../../../data/generated_bad_code'
    
    codes = []
    scores = []
    ids = []
    question_descriptions = []
    count = 0
    
    question_store = '../../../EffiBench/data/dataset.json'
    
    supplement_store = '../../leetcode/trainer/leetcode.json'
    
    with open(question_store, 'r') as f:
        questions = json.load(f)
        
    with open(supplement_store, 'r') as f:
        supplement = json.load(f)
        
    count_no_prompt = 0
    
    prompt_base = open('/workspace/openrhlf_test/prompt_template.txt', 'r').read()
    
    for k, v in test_list.items():
        question_id = k
        for key in v.keys():
            if key == 'reference':
                code = open(os.path.join(code_store, 'reference_human_solution', question_id+'.py')).read().split('solution=Solution()')[0]
                score = v[key]
                codes.append(remove_comments_and_blank_lines(code))
                scores.append(score)
                ids.append(question_id)
        if 'reference' not in v.keys():
            # list as v.keys() 
            key = list(v.keys())[0]
            code = open(os.path.join(code_store, 'crawled_human_solution','crawled_human_solution', question_id, str(key)+'.py')).read().split('solution=Solution()')[0]
            score = v[key]
            if len(remove_comments_and_blank_lines(code)) <= 2000:
                codes.append(remove_comments_and_blank_lines(code))
                scores.append(score)  
                ids.append(question_id)  
            else:
                pass

                
        prompt = prompt_base
        for question in questions:
            if str(question['problem_idx']) == str(question_id):
                # question_descriptions.append(question['markdown_description']+"\n Please write a python class named 'solution' that implements the above requirements.\nThe solution class should be inside ``` and ```.\nPlease only output the solution class without any additional code.\n\n")
                # prompt+='You are an expert Python programmer. Complete the Python class below in the # Solution section to solve the following problem:\n# Problem Description\n'
                # prompt+=question['markdown_description']+"\n#Task Requirement\nPlease complete below Solution class to splve this problem without any additional text.\n\n"
                prompt+=f"\n# Task description:\n```python\n{question['markdown_description']}\n```\n"
                prompt+=f"# Test case:\n```python\n{question['small_test_cases']}\n```"
                prompt+=f"\n\n# Code\n"
                break
            # for question in supplement:
            #     if int(question['id']) == int(question_id):
            #         # question_descriptions.append("```\n"+question['python_template'])
            #         prompt+="# Solution\n```python\n"+question['python_template']
            #         break
                
        # if none of the above, then add a generic prompt
        if "# Solution" not in prompt:
            # prompt+="# Solution\n```python\nclass Solution:\n    def function(self):\n        pass\n```
                count_no_prompt+=1
        question_descriptions.append(prompt)
    
    df = pd.DataFrame({'code': codes, 'score': scores, 'id': ids, 'question_description': question_descriptions})

    ds = CustomDataset(df, tokenizer)

    return ds

def build_dataset_in_dataframe(model_id, test_list):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # load training dataset
    import pickle

    code_store = '../../../data/generated_bad_code'
    
    codes = []
    scores = []
    ids = []
    question_descriptions = []
    count = 0
    
    question_store = '../../../EffiBench/data/dataset.json'
    
    supplement_store = '../../leetcode/trainer/leetcode.json'
    
    with open(question_store, 'r') as f:
        questions = json.load(f)
        
    with open(supplement_store, 'r') as f:
        supplement = json.load(f)
        
    count_no_prompt = 0
    
    prompt_base = open('/workspace/openrhlf_test/prompt_template.txt', 'r').read()
    
    for k, v in test_list.items():
        question_id = k
        for key in v.keys():
            if key == 'reference':
                code = open(os.path.join(code_store, 'reference_human_solution', question_id+'.py')).read().split('solution=Solution()')[0]
                score = v[key]
                codes.append(remove_comments_and_blank_lines(code))
                scores.append(score)
                ids.append(question_id)
        if 'reference' not in v.keys():
            # list as v.keys() 
            key = list(v.keys())[0]
            code = open(os.path.join(code_store, 'crawled_human_solution','crawled_human_solution', question_id, str(key)+'.py')).read().split('solution=Solution()')[0]
            score = v[key]
            if len(remove_comments_and_blank_lines(code)) <= 2000:
                codes.append(remove_comments_and_blank_lines(code))
                scores.append(score)  
                ids.append(question_id)  
            else:
                pass

                
        prompt = prompt_base
        for question in questions:
            if str(question['problem_idx']) == str(question_id):
                # question_descriptions.append(question['markdown_description']+"\n Please write a python class named 'solution' that implements the above requirements.\nThe solution class should be inside ``` and ```.\nPlease only output the solution class without any additional code.\n\n")
                # prompt+='You are an expert Python programmer. Complete the Python class below in the # Solution section to solve the following problem:\n# Problem Description\n'
                # prompt+=question['markdown_description']+"\n#Task Requirement\nPlease complete below Solution class to splve this problem without any additional text.\n\n"
                prompt+=f"\n# Task description:\n```python\n{question['markdown_description']}\n```\n"
                prompt+=f"# Test case:\n```python\n{question['small_test_cases']}\n```"
                prompt+=f"\n\n# Code\n"
                break
            # for question in supplement:
            #     if int(question['id']) == int(question_id):
            #         # question_descriptions.append("```\n"+question['python_template'])
            #         prompt+="# Solution\n```python\n"+question['python_template']
            #         break
                
        # if none of the above, then add a generic prompt
        if "# Solution" not in prompt:
            # prompt+="# Solution\n```python\nclass Solution:\n    def function(self):\n        pass\n```
                count_no_prompt+=1
        question_descriptions.append(prompt)
    
    df = pd.DataFrame({'code': codes, 'score': scores, 'id': ids, 'question_description': question_descriptions})

    return df

def build_emtpy_dataset(model_id, test_list):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # load training dataset
    import pickle

    code_store = '../../../data/generated_bad_code'
    
    codes = []
    scores = []
    ids = []
    question_descriptions = []
    count = 0
    
    question_store = '../../../EffiBench/data/dataset.json'
    
    supplement_store = '../../leetcode/trainer/leetcode.json'
    
    with open(question_store, 'r') as f:
        questions = json.load(f)
        
    with open(supplement_store, 'r') as f:
        supplement = json.load(f)
        
    count_no_prompt = 0
    
    prompt_base = open('/workspace/openrhlf_test/prompt_template.txt', 'r').read()
    
    for k, v in test_list.items():
        question_id = k
        for key in v.keys():
            if key == 'reference':
                code = open(os.path.join(code_store, 'reference_human_solution', question_id+'.py')).read().split('solution=Solution()')[0]
                score = v[key]
                codes.append(remove_comments_and_blank_lines(code))
                scores.append(score)
                ids.append(question_id)
        if 'reference' not in v.keys():
            # list as v.keys() 
            key = list(v.keys())[0]
            code = open(os.path.join(code_store, 'crawled_human_solution','crawled_human_solution', question_id, str(key)+'.py')).read().split('solution=Solution()')[0]
            score = v[key]
            if len(remove_comments_and_blank_lines(code)) <= 2000:
                codes.append(remove_comments_and_blank_lines(code))
                scores.append(score)  
                ids.append(question_id)  
            else:
                pass

                
        prompt = prompt_base
        for question in questions:
            if str(question['problem_idx']) == str(question_id):
                # question_descriptions.append(question['markdown_description']+"\n Please write a python class named 'solution' that implements the above requirements.\nThe solution class should be inside ``` and ```.\nPlease only output the solution class without any additional code.\n\n")
                # prompt+='You are an expert Python programmer. Complete the Python class below in the # Solution section to solve the following problem:\n# Problem Description\n'
                # prompt+=question['markdown_description']+"\n#Task Requirement\nPlease complete below Solution class to splve this problem without any additional text.\n\n"
                prompt+=f"\n# Task description:\n```python\n{question['markdown_description']}\n```\n"
                prompt+=f"# Test case:\n```python\n{question['small_test_cases']}\n```"
                prompt+=f"\n\n# Code\n```python"
                break
            # for question in supplement:
            #     if int(question['id']) == int(question_id):
            #         # question_descriptions.append("```\n"+question['python_template'])
            #         prompt+="# Solution\n```python\n"+question['python_template']
            #         break
                
        # if none of the above, then add a generic prompt
        if "# Solution" not in prompt:
            # prompt+="# Solution\n```python\nclass Solution:\n    def function(self):\n        pass\n```
                count_no_prompt+=1
        question_descriptions.append(prompt)
    
    df = pd.DataFrame({'code': codes, 'score': scores, 'id': ids, 'question_description': ''})

    ds = CustomDataset(df, tokenizer)

    return ds