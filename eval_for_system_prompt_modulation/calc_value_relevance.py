import pandas as pd
from collections import Counter
import os
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser(description='generate two sets of system prompts for steering to supporting and opposing values for principle')

load_dotenv()

parser.add_argument("--input_system_prompt_file", "-i")
parser.add_argument("--output_jsonl_file_div","-o")

args = parser.parse_args()

input_file_div = args.input_system_prompt_file
basename = os.path.splitext(os.path.basename(input_file_div))[0]
output_file_div = f'{args.output_jsonl_file_div}/{basename}_clean.csv'

values_df = pd.read_csv('/Users/kellycyy/Desktop/moral dilemma/analysis_after_topic_model/analysis_on_value_with_llm_training/values.csv')
values = set(values_df['value'])

def clean_eval(text):
    output = text.split('Supporting values:')[-1]
    sup_values = output.split('Opposing values: ')[0]
    opp_values = output.split('Opposing values: ')[-1]

    sup_values_clean =  sup_values.split('[')[-1].split(']')[0].split(',')
    opp_values_clean =  opp_values.split('[')[-1].split(']')[0].split(',')

    sup_values_list = [i.strip() for i in sup_values_clean]
    opp_values_list = [i.strip() for i in opp_values_clean]
    return sup_values_list, opp_values_list


df = pd.read_csv(input_file_div, index_col=0)
row_list = []
for i, row in df.iterrows():
    total_sup_values_list = []
    total_opp_values_list = []
    for i in range(10):
        sup_values_list, opp_values_list = clean_eval(row[f'model_resp_eval_{i}'])
        total_sup_values_list.append(sup_values_list)
        total_opp_values_list.append(opp_values_list)

    flatten_sup_values_list = [x for xs in total_sup_values_list for x in xs]
    flatten_opp_values_list = [x for xs in total_opp_values_list for x in xs]
    counter_sup_value = Counter(flatten_sup_values_list)
    counter_opp_value = Counter(flatten_opp_values_list)
    row['counter_sup_value'] = counter_sup_value
    row['counter_opp_value'] = counter_opp_value
    sup_value_prob_dict = {k: v/10 for k,v in counter_sup_value.items() if k in values}
    opp_value_prob_dict = {k: v/10 for k,v in counter_opp_value.items() if k in values}
    row['sup_value_prob_dict'] = sup_value_prob_dict
    row['opp_value_prob_dict'] = opp_value_prob_dict

    row_list.append(row)

df_clean = pd.DataFrame(row_list)
df_clean.to_csv(output_file_div)