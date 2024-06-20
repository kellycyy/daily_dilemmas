import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description='get values conflict for principle')
parser.add_argument("--input_principle_file","-i")
parser.add_argument("--input_dilemma_file","-d")
parser.add_argument("--output_jsonl_file_div","-o")
parser.add_argument("--model","-m")
parser.add_argument("--api_key", required=False, help="API key for the service. Can also be set via the API_KEY environment variable.")
args = parser.parse_args()

api_key = args.api_key or os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key is required. Set it through argument or as an environment variable or in a .env file.")
openai_api_key = api_key

model = args.model

input_principle_file = args.input_principle_file
input_dilemma_file = args.input_dilemma_file
output_eval_json_file = f'{args.output_jsonl_file_div}//principle_{model}_eval_with_values_conflicts_and_dilemma.jsonl'
output_eval_csv_file = f'{args.output_jsonl_file_div}//principle_{model}_eval_with_values_conflicts_and_dilemma.csv'

principle_df = pd.read_csv(input_principle_file)
dilemma_df = pd.read_csv(input_dilemma_file)
values_df = pd.read_csv('/Users/kellycyy/Desktop/moral dilemma/for_supplementary/moral_dilemma/data/values.csv', index_col=0)

def form_dict(filtered_df_excluded, model_resp, value_relevant):
    data_dict_idx = defaultdict(lambda: defaultdict(list))
    data_dict_cnt = defaultdict(lambda: defaultdict(int))
    for idx, row in filtered_df_excluded.iterrows():
        values = row['values_names']
        if value_relevant in values:
            if row['action_type'] == row[model_resp]:
                data_dict_idx[value_relevant]['selected'].append(row['idx'])
                data_dict_cnt[value_relevant]['selected'] += 1
            else:
                data_dict_idx[value_relevant]['neglected'].append(row['idx'])
                data_dict_cnt[value_relevant]['neglected'] += 1 
    return data_dict_idx, data_dict_cnt

filtered_df = dilemma_df[(dilemma_df[f'model_resp_{model}_clean'] == 'not_to_do') | (dilemma_df[f'model_resp_{model}_clean'] == 'to_do')]

rows_list = []
for i, row in tqdm(principle_df.iterrows(), total = len(principle_df)):
    data_dict = row.to_dict()
    sup_value_prob_dict_old = eval(row['sup_value_prob_dict'])
    opp_value_prob_dict_old = eval(row['opp_value_prob_dict'])
    values_from_value_df =  set(values_df['value'])
    sup_value_prob_dict = dict((key,sup_value_prob_dict_old[key]) for key in sup_value_prob_dict_old if key in values_from_value_df)
    opp_value_prob_dict = dict((key,opp_value_prob_dict_old[key]) for key in opp_value_prob_dict_old if key in values_from_value_df)
    row['sup_value_prob_dict_clean'] = sup_value_prob_dict
    row['opp_value_prob_dict_clean'] = opp_value_prob_dict
    data_dict_idx_list, data_dict_cnt_list = [], []
    
    for k, v in sup_value_prob_dict.items():
        data_dict_idx, data_dict_cnt = form_dict(filtered_df, f'model_resp_{model}_clean', k)
        data_dict_idx_list.append(data_dict_idx)
        data_dict_cnt_list.append(data_dict_cnt)
        data_dict_idx_final = dict((key,d[key]) for d in data_dict_idx_list for key in d)
        data_dict_cnt_final = dict((key,d[key]) for d in data_dict_cnt_list for key in d)
        
    row['sup_values_data_dict_cnt'] = data_dict_cnt_final
    row['sup_values_data_dict_idx'] = data_dict_idx_final

    sup_values_data_dict_idx_list = []
    for k, d in data_dict_idx_final.items():
        for i, j in d.items():
            for item in j:
                sup_values_data_dict_idx_list.append(item)

    sup_values_data_dict_idx_set = set(sup_values_data_dict_idx_list)
    data_dict_prob = {}
    for k, v in sup_value_prob_dict.items():
        data_dict_prob[k] = sup_value_prob_dict[k]*(data_dict_cnt_final[k]['selected']+data_dict_cnt_final[k]['neglected'])
    row['sup_values_calculate_with_prob'] = data_dict_prob
    
    data_dict_idx_list, data_dict_cnt_list = [], []
    for k, v in opp_value_prob_dict.items():
        data_dict_idx, data_dict_cnt = form_dict(filtered_df, f'model_resp_{model}_clean', k)
        data_dict_idx_list.append(data_dict_idx)
        data_dict_cnt_list.append(data_dict_cnt)
        data_dict_idx_final = dict((key,d[key]) for d in data_dict_idx_list for key in d)
        data_dict_cnt_final = dict((key,d[key]) for d in data_dict_cnt_list for key in d)
    row['opp_values_data_dict_cnt'] = data_dict_cnt_final
    row['opp_values_data_dict_idx'] = data_dict_idx_final
    opp_values_data_dict_idx_set = set(data_dict_idx_final)


    opp_values_data_dict_idx_list = []
    for k, d in data_dict_idx_final.items():
        for i, j in d.items():
            for item in j:
                opp_values_data_dict_idx_list.append(item)

    opp_values_data_dict_idx_set = set(opp_values_data_dict_idx_list)

    data_dict_prob = {}
    for k, v in opp_value_prob_dict.items():
        data_dict_prob[k] = opp_value_prob_dict[k]*(data_dict_cnt_final[k]['selected']+data_dict_cnt_final[k]['neglected'])

    row['opp_values_calculate_with_prob'] = data_dict_prob
    sup_values_calculate_with_prob_dict = row['sup_values_calculate_with_prob']
    opp_values_calculate_with_prob_dict = row['opp_values_calculate_with_prob']
    row['total_score_by_sup_values_calculate_with_prob'] = sum(sup_values_calculate_with_prob_dict.values())
    row['total_score_by_opp_values_calculate_with_prob'] = sum(opp_values_calculate_with_prob_dict.values())
    row['combined_score_by_sup_opp_values_calculate_with_prob'] = row['total_score_by_sup_values_calculate_with_prob'] - row['total_score_by_opp_values_calculate_with_prob']

    intercept_set = sup_values_data_dict_idx_set.intersection(opp_values_data_dict_idx_set)
    # row['conflict_values_idx_intercept'] = intercept_set

    real_conflict_dilemma = []
    for index in intercept_set:
        filter_dilemma_df = dilemma_df[dilemma_df['idx'] == index]
        indicator_real_conflict = 0
        dilemma_types_for_sup_and_opp = []
        for idx, innerrow in filter_dilemma_df.iterrows():
            values = eval(innerrow['values_names'])
            values_types_for_sup_and_opp = []
            for value in values:
                sup_values = [k for k, v in sup_value_prob_dict.items()]
                opp_values = [k for k, v in opp_value_prob_dict.items()]
                if value in sup_values:
                    values_types_for_sup_and_opp.append('sup')
                if value in opp_values:
                    values_types_for_sup_and_opp.append('opp')
            for value_type in list(set(values_types_for_sup_and_opp)):
                dilemma_types_for_sup_and_opp.append(value_type)
            if len(set(values_types_for_sup_and_opp)) != 1:
                indicator_real_conflict = 1
        if len(set(dilemma_types_for_sup_and_opp)) == 1:
            indicator_real_conflict = 1
        if indicator_real_conflict == 0:
            real_conflict_dilemma.append(index)

    row['real_conflict_dilemma_idx'] = real_conflict_dilemma

## new  
    data_dict_idx_list, data_dict_cnt_list = [], []
    filter_dilemma_df = dilemma_df[dilemma_df['idx'].isin(real_conflict_dilemma)]

    for k, v in sup_value_prob_dict.items():
        data_dict_idx, data_dict_cnt = form_dict(filter_dilemma_df, f'model_resp_{model}_clean', k)
        data_dict_idx_list.append(data_dict_idx)
        data_dict_cnt_list.append(data_dict_cnt)
        data_dict_idx_final = dict((key,d[key]) for d in data_dict_idx_list for key in d)
        data_dict_cnt_final = dict((key,d[key]) for d in data_dict_cnt_list for key in d)
        
        
    sup_values_data_dict_idx_list = []
    for k, d in data_dict_idx_final.items():
        for i, j in d.items():
            for item in j:
                sup_values_data_dict_idx_list.append(item)

    row['dilemma_sup_values_data_dict_cnt'] = data_dict_cnt_final
    row['dilemma_sup_values_data_dict_idx'] = data_dict_idx_final
    sup_values_data_dict_idx_set = set(sup_values_data_dict_idx_list)
    data_dict_prob = {}
    for k, v in sup_value_prob_dict.items():
        if k in data_dict_cnt_final:
            data_dict_prob[k] = sup_value_prob_dict[k]*(data_dict_cnt_final[k]['selected']+data_dict_cnt_final[k]['neglected'])

    row['dilemma_sup_values_calculate_with_prob'] = data_dict_prob
    
    data_dict_idx_list, data_dict_cnt_list = [], []
    for k, v in opp_value_prob_dict.items():
        data_dict_idx, data_dict_cnt = form_dict(filter_dilemma_df, f'model_resp_{model}_clean', k)
        data_dict_idx_list.append(data_dict_idx)
        data_dict_cnt_list.append(data_dict_cnt)
        data_dict_idx_final = dict((key,d[key]) for d in data_dict_idx_list for key in d)
        data_dict_cnt_final = dict((key,d[key]) for d in data_dict_cnt_list for key in d)
    row['dilemma_opp_values_data_dict_cnt'] = data_dict_cnt_final
    row['dilemma_opp_values_data_dict_idx'] = data_dict_idx_final
    opp_values_data_dict_idx_set = set(data_dict_idx_final)


    opp_values_data_dict_idx_list = []
    for k, d in data_dict_idx_final.items():
        for i, j in d.items():
            for item in j:
                opp_values_data_dict_idx_list.append(item)

    opp_values_data_dict_idx_set = set(opp_values_data_dict_idx_list)

    data_dict_prob = {}
    for k, v in opp_value_prob_dict.items():
        if k in data_dict_cnt_final:
            data_dict_prob[k] = opp_value_prob_dict[k]*(data_dict_cnt_final[k]['selected']+data_dict_cnt_final[k]['neglected'])

    row['dilemma_opp_values_calculate_with_prob'] = data_dict_prob
    sup_values_calculate_with_prob_dict = row['dilemma_sup_values_calculate_with_prob']
    opp_values_calculate_with_prob_dict = row['dilemma_opp_values_calculate_with_prob']
    row['dilemma_total_score_by_sup_values_calculate_with_prob'] = sum(sup_values_calculate_with_prob_dict.values())
    row['dilemma_total_score_by_opp_values_calculate_with_prob'] = sum(opp_values_calculate_with_prob_dict.values())
    row['dilemma_combined_score_by_sup_opp_values_calculate_with_prob'] = row['dilemma_total_score_by_sup_values_calculate_with_prob'] - row['dilemma_total_score_by_opp_values_calculate_with_prob']

    rows_list.append(row)

    
df_final = pd.DataFrame(rows_list)
df_final.to_csv(f'{output_eval_csv_file}')