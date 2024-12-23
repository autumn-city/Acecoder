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
