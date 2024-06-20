import os 
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='combine model response file to get values and five theories details')
parser.add_argument("--input_model_resp_file","-i")
parser.add_argument("--input_dilemma_file","-dilemma")
parser.add_argument("--output_analysis_file","-o")
parser.add_argument("--model","-m")

args = parser.parse_args()

dilemma_action_separate_with_value_detail_file_path = args.input_dilemma_file
model_resp_file = args.input_model_resp_file
model = args.model
output_file = f'{args.output_analysis_file}/{model}_eval_on_dilemmas_with_action_separate.csv'

df_model_resp = pd.read_csv(model_resp_file)
df = pd.read_csv(dilemma_action_separate_with_value_detail_file_path)

new_data = []
for i, row in tqdm(df.iterrows()):
    index = row['idx']
    data = row.to_dict()
    model_resp_eval = df_model_resp[df_model_resp['idx'] == index][f'model_resp_{model}_clean'].values[0]
    data[f'model_resp_{model}_clean'] = model_resp_eval
    new_data.append(data)

df_new = pd.DataFrame(new_data)
df_new.to_csv(output_file)