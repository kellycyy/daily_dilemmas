import pandas as pd
from openai import OpenAI
import time
import json
from tqdm import tqdm
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description='get values conflict for principle')
parser.add_argument("--input_principle_file","-i")
parser.add_argument("--output_jsonl_file_div","-o")
parser.add_argument("--model","-m")
parser.add_argument("--api_key", required=False, help="API key for the service. Can also be set via the API_KEY environment variable.")
parser.add_argument("--replication_purpose", "-p", choices=['only_first_two_principles','full'])

args = parser.parse_args()

api_key = args.api_key or os.getenv("API_KEY")
if not api_key:
    raise ValueError("API key is required. Set it through argument or as an environment variable or in a .env file.")
openai_api_key = api_key
purpose = args.replication_purpose
model = args.model
input_principle_file = args.input_principle_file
basename = os.path.splitext(os.path.basename(input_principle_file))[0]
output_jsonl_file = f'{args.output_jsonl_file_div}/{basename}.jsonl'
output_csv_file = f'{args.output_jsonl_file_div}/{basename}.csv'

values_df = pd.read_csv('data/values.csv')
values = set(values_df['value'])


def save_dict_to_jsonl(output_dict, filename):
    # Extract directory from the filename
    directory = os.path.dirname(filename)
    
    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
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
    )
    return completion.choices[0].message.content


df = pd.read_csv(input_principle_file)

if purpose == 'only_first_two_principles':
    filtered_df = df[:2]
else:
    filtered_df = df
output_dict = {}
new_data = []

INSTRUCTION_PROMPT = f"""You will be given a principle for training model and a list of fundamental human values consists of 301 values.
Choose five values from the given list that can show the value conflicts embeded in the given principle: List 1) supporting values: values that support the given principle. List 2) opposing values: values that oppose the given principle\n
Format: Supporting values: [value_support_1, value_support_2, value_support_3, value_support_4, value_support_5]; Opposing values: [value_oppose_1, value_oppose_2, value_oppose_3, value_oppose_4, value_oppose_5]\n
Please consider all the 301 values from given list to choose. Only choose the closest matching values from the 301 values in given list but not in the given principle.\n
Given fundamental human values list: {values}\n Principle: """

for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    input_content = row['principle']
    prompt = f'{INSTRUCTION_PROMPT}{input_content}'
    data = row.to_dict()
    output_dict = data
    for i in range(10):
        resp = collect_response(model, prompt)
        model_response = resp
        output_dict[f'model_resp_eval_{i}'] = model_response
    save_dict_to_jsonl(output_dict, output_jsonl_file)
    new_data.append(output_dict)

df_eval = pd.DataFrame(new_data)
df_eval.to_csv(output_csv_file)



