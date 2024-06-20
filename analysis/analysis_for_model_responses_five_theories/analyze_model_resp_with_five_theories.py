import math
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='analyze model response file with five theories')
parser.add_argument("--input_model_resp_file", "-i")
parser.add_argument("--output_analysis_file", "-o")
parser.add_argument("--models", "-m", nargs='+')
args = parser.parse_args()

input_file_path = args.input_model_resp_file
output_file_path = args.output_analysis_file
models = args.models

df = pd.read_csv(input_file_path)
metrics = ['wvs','mft','virtue','emotion','maslow']
nan = math.nan

def calculate_metric_difference(counter1, counter2):
    result = counter1.copy()
    for key, value in counter2.items():
        if key in result:
            result[key] -= value
        else:
            result[key] = 0
            result[key] -= value
    return result

def divide_dict_by_int_by_prop(dictionary, divisor_dict):
    divided_dict = {}
    for key, value in dictionary.items():
        divisor = divisor_dict[key]
        divided_dict[key] = (value / divisor)*100
    return divided_dict

def form_two_dicts(filtered_df_excluded, model_resp):
    data_selected = defaultdict(list)
    data_neglected = defaultdict(list)
    for idx, row in filtered_df_excluded.iterrows():
        if row['action_type'] == row[model_resp]:
            # If it matches, append the index to the 'selected' 
            data_selected['idx'].append(row['idx'])
            for metric in metrics:
                data_selected[metric].append(eval(row[metric]))
        else:
            # If it does not match, append the index to the 'neglected' list
            data_neglected['idx'].append(row['idx'])
            for metric in metrics:
                data_neglected[metric].append(eval(row[metric]))
    return data_selected, data_neglected

def remove_nan_keys_from_counter(counter):
    for key in list(counter.keys()):
        if isinstance(key, float) and math.isnan(key):
            del counter[key]
    return counter

def clean_counter_for_metric(data_action, metric):
    data_action_metric = sum(data_action[metric], [])
    data_action_metric_counter = Counter(data_action_metric)
    data_action_metric_counter = remove_nan_keys_from_counter(data_action_metric_counter)
    return data_action_metric_counter


def clean_and_aggregate_counters(data_selected, data_neglected, metric):
    selected_counter = clean_counter_for_metric(data_selected, metric)
    neglected_counter = clean_counter_for_metric(data_neglected, metric)
    total_counter = selected_counter + neglected_counter
    return selected_counter, neglected_counter, total_counter

def normalize_counters(selected_counter, neglected_counter, total_counter):
    # Normalize the counters by proportion
    normalized_selected = divide_dict_by_int_by_prop(selected_counter, total_counter)
    normalized_neglected = divide_dict_by_int_by_prop(neglected_counter, total_counter)
    return normalized_selected, normalized_neglected
 
total_metric_counter_dict_for_diff_models_list = []
for model in models:
    filtered_df = df[(df[f'model_resp_{model}_clean'] == 'not_to_do') | (df[f'model_resp_{model}_clean'] == 'to_do')]
    data_selected, data_neglected = form_two_dicts(filtered_df, f'model_resp_{model}_clean')
    total_metric_counter_dict = dict()
    total_metric_counter_dict['model'] = model
    for metric in tqdm(metrics):
        selected_counter, neglected_counter, total_counter = clean_and_aggregate_counters(data_selected, data_neglected, metric)
        normalized_selected, normalized_neglected = normalize_counters(selected_counter, neglected_counter, total_counter)
        metric_diff = calculate_metric_difference(normalized_selected, normalized_neglected)
        total_metric_counter_dict[metric] = metric_diff
    total_metric_counter_dict_for_diff_models_list.append(total_metric_counter_dict)
output_df = pd.DataFrame(total_metric_counter_dict_for_diff_models_list)
output_df.to_csv(output_file_path)