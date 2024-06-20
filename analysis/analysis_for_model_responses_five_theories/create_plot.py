import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
import argparse
import argparse
import json

parser = argparse.ArgumentParser(description='Create plot from analysis model resp file on five theories')
parser.add_argument("--input_analysis_file", '-i')
parser.add_argument("--output_graph_div", '-o')

args = parser.parse_args()

input_file_path = args.input_analysis_file
output_file_path = args.output_graph_div

config_path = 'analysis/analysis_for_model_responses_five_theories/config.json'
with open(config_path, 'r') as config_file:
    format_for_metric = json.load(config_file)

metrics = ['wvs','mft','virtue','emotion','maslow']

def plot_barh_multiple_dicts(dicts, labels, bar_width, output_file_path):
    num_models = len(labels)
    colors = ['#8c564b',  '#d62728', '#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c']
    keys = format_for_metric[metric]['labels']
    num_dicts = len(dicts)
    if num_models == 1:
        bar_width = 0.4
    else:
        bar_width = min(0.8 / num_models, 0.1) 

    r = [np.arange(len(keys)) + i * bar_width for i in range(num_dicts)]

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    min_value = float('inf')
    max_value = float('-inf')

    for i, d in enumerate(dicts):
        values = [d.get(key, 0) for key in keys]  # Get the values
        min_value = min(min_value, min(values))
        max_value = max(max_value, max(values))
        plt.barh(r[i], values, height=bar_width, edgecolor='grey', color=colors[i], label=labels[i])

    keys = format_for_metric[metric]['new_labels']
    plt.yticks(r[0] + bar_width * (num_dicts - 1) / 2, keys, fontsize=22)
    
    title_name = format_for_metric[metric]['name']
    plt.title(f'{title_name}', fontsize=32, weight='bold')
    plt.legend(fontsize='small', ncol=num_models, loc='upper center', bbox_to_anchor=(0.5, -0.05),)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    current_ticks = plt.gca().get_xticks()

    x_min = max(-100, max(min_value, -100)-10)
    x_max = min(100, min(max_value, 100)+10)
    plt.xlim(x_min, x_max)

    new_ticks = [tick for tick in current_ticks if x_min <= tick <= x_max]
    plt.xticks(new_ticks, [f'{int(tick)}%' for tick in new_ticks], fontsize=15)

    plt.tight_layout()
    plt.savefig(output_file_path)


analysis_df = pd.read_csv(input_file_path)
models = list(analysis_df['model'])

for metric in tqdm(metrics):
    total_counter_list = [eval(one_dict) for one_dict in list(analysis_df[metric])]
    plot_barh_multiple_dicts(total_counter_list, models, 0.07, f'{output_file_path}/percentage_diff_prop_{metric}.jpg')
