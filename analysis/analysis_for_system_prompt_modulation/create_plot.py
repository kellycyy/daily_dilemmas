import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser(description='evaluate models steerabiltiy by system prompt')

load_dotenv()
parser.add_argument("--input_support_value_file", "-sup")
parser.add_argument("--input_oppose_value_file", "-opp")
parser.add_argument("--output_graph_div", "-o")
parser.add_argument("--replication_purpose", "-p", choices=['only_two_principles','full'])
args = parser.parse_args()

input_sup_file = args.input_support_value_file
input_opp_file = args.input_oppose_value_file
output_graph_div = args.output_graph_div
purpose = args.replication_purpose
I_TH_TIME = 9

def insert_line_breaks(label, every_n_words=2):
    if ':' in label:
        new_label = f"Exception:\nTransformation\ntasks"
    elif 'information' in label:
        new_label = f"Don't provide\ninformation\nhazard"
    elif 'creator' in label:
        new_label = f"Respect\ncreators and\ntheir rights"
    elif 'fairness' in label:
        new_label = f"Encourage\nfaireness and\nkindness and\ndiscourage hate"
    elif 'uncertainty' in label:
        new_label = f"Express\nuncertainty"
    elif 'privacy' in label:
        new_label = f"Protect\nPeople's\nPrivacy"
    else:
        words = label.split(' ')
        new_label = '\n'.join(' '.join(words[i:i+every_n_words]) for i in range(0, len(words), every_n_words))

    return new_label


def plot_score(dfa, dfb, plot_item,i_th_time):
    df_merged = pd.merge(dfa, dfb, on='principle')

    df_long = pd.melt(df_merged, id_vars=['principle'], value_vars=[f'{plot_item}_original', f'{plot_item}_sup', f'{plot_item}_opp'])

    df_long['Field'] = df_long['variable'].apply(lambda x: 'score' if 'score' in x else 'win_rate')
    df_long['Type'] = df_long['variable'].apply(lambda x: 'Before modulation' if 'original' in x else ('After modulation towards supportive values' if 'sup' in x else 'After modulation towards opposing values'))

    principles = df_long['principle'].unique()
    principles_line_break = [insert_line_breaks(principle) for principle in principles]

    index_map = {principle: i for i, principle in enumerate(principles)}
    indexed_labels = [f"{index_map[principle]+i_th_time}\n{label}" for principle, label in zip(principles, principles_line_break)]

    df_long['x'] = df_long.apply(lambda row: index_map[row['principle']] + (0 if row['Field'] == plot_item else 1), axis=1)

    df_long['x_label'] = df_long.apply(lambda row: insert_line_breaks(f"{row['principle']} - {row['Field']}"), axis=1)

    color_map = {
        'score_original': '#d7642c',
        'score_sup': '#eba532',
        'win_rate_sup': '#41afaa',
        'win_rate_original': '#466eb4',
        'score_opp': '#af4b91',
        'win_rate_opp': '#00a0e1'
    }

    markers = ['o', '^', 'v', 'o', '^', 'v']

    plt.figure(figsize=(23, 12))
    
    sns.pointplot(data=df_long, x='x', y='value', hue='variable', palette=color_map, dodge=0, join=False, markers=markers,  scale = 2.6)

    principles_line_break = [insert_line_breaks(i) for i in principles]
    
    plt.xticks(ticks=[index_map[principle] for principle in principles], labels=indexed_labels, rotation=0, fontsize=24)
    plt.yticks(fontsize=25)
    plt.xlabel('Principles', fontsize=30)
    if plot_item == 'win_rate':
        ylabel = 'Win Rate (%)'
    else:
        ylabel = 'Weighted Score Diff.'
    plt.ylabel(f'{ylabel}', fontsize=35)

    legend_markers = {
        f'{plot_item}_original': 'o',
        f'{plot_item}_sup': '^',
        f'{plot_item}_opp': 'v'
    }
    
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_labels = {
        f'{plot_item}_original': f'Before modulation',
        f'{plot_item}_sup': f'After modulation\n(supportive values)',
        f'{plot_item}_opp': f'After modulation\n(opposing values)'
    }
    legend_labels = [legend_labels[label] for label in labels]
    plt.legend(handles, legend_labels,loc='upper left', fontsize=28, title_fontsize=20)

    for principle in principles[:-1]:
        plt.axvline(x=index_map[principle] + 0.5, color='gray', linestyle='--')
    plt.axhline(0, linestyle='--', linewidth=2, color='grey')
    plt.ylim(-10, 15)
    plt.tight_layout(pad=1)

    plt.savefig(f'{output_graph_div}/{plot_item}_start_from_{i_th_time}.png', format='png', bbox_inches='tight', dpi=300)  # Reduce border in saved figure


df_sup = pd.read_csv(input_sup_file)
df_opp = pd.read_csv(input_opp_file)

dfa = df_sup[['principle']]
dfb = df_opp[['principle']]

dfa['score_original'] =  df_sup['dilemma_combined_score_by_sup_opp_values_calculate_with_prob']
dfa['score_sup'] = df_sup['system_prompt_sup_value_system_prompt_dilemma_combined_score_by_sup_opp_values_calculate_with_prob']
dfb['score_opp'] = df_opp['system_prompt_opp_value_system_prompt_dilemma_combined_score_by_sup_opp_values_calculate_with_prob']


if purpose == 'full':
    plot_score(dfa[:I_TH_TIME], dfb[:I_TH_TIME],'score',0)
    plot_score(dfa[I_TH_TIME:], dfb[I_TH_TIME:],'score',I_TH_TIME)
else:
    plot_score(dfa, dfb,'score',0)