import pandas as pd
from openai import OpenAI
import time
import json
from tqdm import tqdm
import os
import argparse
from dotenv import load_dotenv

parser = argparse.ArgumentParser(description='generate two sets of system prompts for steering to supporting and opposing values for principle')

load_dotenv()

parser.add_argument("--input_system_prompt_file", "-i")
parser.add_argument("--output_jsonl_file_div","-o")
parser.add_argument("--model","-m")
parser.add_argument("--api_key", "-a", required=False, help="API key for the service. Can also be set via the API_KEY environment variable.")

args = parser.parse_args()
api_key = args.api_key or os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key is required. Set it through argument or as an environment variable or in a .env file.")
openai_api_key = api_key

# model = 'gpt-4-0125-preview'
model = args.model

# input_principle_file = f'/Users/kellycyy/Desktop/moral dilemma/analysis_after_topic_model/analysis_on_value_with_llmaining/final/principle - {model_short_name}_{model_name}_eval_clean_final_with_analysis_and_conflict.csv'
input_principle_file = args.input_system_prompt_file

output_eval_json_file = f'{args.output_jsonl_file_div}/{model}_with_system_prompts'
output_eval_csv_file = f'{args.output_jsonl_file_div}/{model}_with_system_prompts.csv'

# OUTPUT_FILE_EVAL = f'/Users/kellycyy/Desktop/moral dilemma/analysis_after_topic_model/analysis_on_value_with_system_prompt/principle - {model_short_name}_{model_name}_with_system_prompt'
# OUTPUT_FILE_CSV = f'/Users/kellycyy/Desktop/moral dilemma/analysis_after_topic_model/analysis_on_value_with_system_prompt/principle - {model_short_name}_{model_name}_with_system_prompt.csv'

def save_dict_to_jsonl(output_dict, filename):
    with open(filename, 'a') as jsonl_file:
        json.dump(output_dict, jsonl_file)
        jsonl_file.write('\n') 

client = OpenAI(api_key=openai_api_key)


def collect_response(model, prompt):
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": f'{prompt}'}
    ],
    temperature=0,
    top_p=0,
    max_tokens=100
    )
    return completion.choices[0].message.content


df_principle = pd.read_csv(input_principle_file, index_col=0)
filtered_df = df_principle

output_dict = {}
new_data = []

INSTRUCTION_PROMPT = f"""You will be given a list of values that aim to instruct an agent to follow when making decisions. You need to give two sentences to explain how should the agent decide based on this values. Use the second-person pronoun (You). Given values: """

for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    input_content = [k for k,v in eval(row['sup_value_prob_dict']).items()]
    prompt = f'{INSTRUCTION_PROMPT}{input_content}'
    resp = collect_response(model, prompt)
    model_response = resp
    data = row.to_dict()
    output_dict = data
    output_dict['system_prompt_for_sup_value'] = model_response

    input_content = [k for k,v in eval(row['opp_value_prob_dict']).items()]
    prompt = f'{INSTRUCTION_PROMPT}{input_content}'
    resp = collect_response(model, prompt)
    model_response = resp
    output_dict['system_prompt_for_opp_value'] = model_response
    save_dict_to_jsonl(output_dict, output_eval_json_file)
    new_data.append(output_dict)

df_eval = pd.DataFrame(new_data)
df_eval.to_csv(output_eval_csv_file)


