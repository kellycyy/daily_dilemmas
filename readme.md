# MoralDilemmas 

**Dataset**: ```data/dilemmas_with_detail_by_action.csv```

**Croissant metadata**: ```metadata/crossiant```

**Three use cases** -- 1. evaluate model preferences on everyday dilemmas 2. evaluate model preference on dilemmas based on the given principle 3. evaluate the model's steerability by system prompt modulation on each given principle

We used ``gpt-4-turbo`` as demonstrated below. We also provided our model responses on our dilemmas dataset run by five models, and some corresponding analysis datasets for replication purposes to avoid running them again through the flag (``replication_purpose``).

## Setup the environment
```
$ conda create --name <env> --file requirements.txt
```

## Case 1: Evaluate model preferences on everyday dilemmas
### 1. Evaluate the gpt-4-turbo model on our dilemma dataset.
```
python eval/evaluate_model_on_dilemma.py \
--output_jsonl_file_div <output_jsonl_file_div> \ 
--model <gpt-4-turbo> \
--model_name_for_variable <own_created_name_e.g.gpt4> \
--replication_purpose <full/only_first_five> \
--api_key <optional_you_can_set_as_.env_or_global>
```
**Required Arguments with suggestions:**
1. ``output_jsonl_file_div``: ``eval/model_response_from_eval``. 
2. ``model name``: `gpt-4-turbo`.
3. ``replicaiton_purpose``: [``full``, ``only_first_five``]: [full dilemmas dataset, only first five dilemmas]

**Optional Arguments:**
1. ``api_key``: you can input OpenAI API key as an argument. Use a .env file or set as a global variable

### 2. Combine model response from 1 to get the details of the dilemma for each action. 
```
python eval/combine_model_eval_with_dilemmas_action_separate.py \
--input_model_resp_file <step_1_output_jsonl_file> \
--input_dilemma_file <MoralDilemma_dataset> \
--output_analysis_file <output_analysis_file_div> \
--model <model_name_for_variable>
```
**Required Arguments with suggestions:**
1. ``input_model_resp_file``: `eval/model_response_from_eval`
2. ``input_dilemma_file``: `data/dilemmas_with_detail_by_action.csv` or `data/dilemmas_with_detail_by_action_test.csv` if use only first five data in step 1
3. ``output_analysis_file``: `eval/model_response_from_eval`. The data of `<model_name>_eval_on_dilemmas_with_action_separate.csv` will be created.

### 3. Analyze the combined model responses file with dilemmas details from step 2 based on the five theories.
```
python analysis/analysis_for_model_responses_five_theories/analyze_model_resp_with_five_theories.py \
--input_model_resp_file <output_from_step_2> \
--output_analysis_file <e.g.analysis/analysis_for_model_responses_five_theories/model_resp.csv> \
--models <model_names>
```
**Required Arguments with suggestions:**
1. ``input_model_resp_file``: use output file from step 2. e.g., ``eval/model_response_from_eval/<model_name>_eval_on_dilemmas_with_action_separate.csv``. We also provided the six model responses in ```eval/model_response_from_eval/all_models_eval_on_dilemmas_with_detail_by_action.csv```
2. ``models``: ``gpt4``; For six models, ``gpt4 gpt35 llama2 llama3 mixtral_rerun claude``

### 4. Plot the graphs from the analyzed files with five theories dimensions.
```
python analysis/analysis_for_model_responses_five_theories/create_plot.py \
--input_analysis_file <output_path_in_step_3/model_resp.csv> \
--output_graph_div <path_for_graphs>
```
**Required Arguments with suggestions:**
1. ``input_analysis_file``: Output from step 3 e.g., ``analysis/analysis_for_model_responses_five_theories/``

## Case 2: Evaluate model preferences on everyday dilemmas based on the given principle

### 1. Get the relevant values for each principle from collected values by prompting the model 10 times.
```
python eval_for_system_prompt_modulation/get_values_for_principle.py \
--input_principle_file <data/principle_company.csv> \
--output_jsonl_file_div <e.g._eval_for_system_prompt_modulation/values_and_system_prompt_for_principle> \
--model <openai_model_e.g._gpt-4-0125-preview> \
--replication_purpose <full_/or/_only_first_two_principles>\
--api_key <optional_you_can_set_as_.env_or_global>
```
**Required Arguments with suggestions:**
1. ``input_principle_file``: ``data/principle_openai.csv``
2. ``model``: ``gpt-4-0125-preview``
3. ``replicaiton_purpose``: [``full``, ``only_first_two_principles``]
4. 
**Optional Arguments:**
1. ``api_key``: you can input OpenAI API key as an argument. Use a .env file or set as a global variable

<!-- - To provide a value list, we used 301 values with the five theories dimensions data as default: ```data/values.csv``` -->

### 2. Calculate the value's relevance by the empirical probabilities from the 10 model responses in step 1.
```
python eval_for_system_prompt_modulation/calc_value_relevance.py \
--input_system_prompt_file <output_from_step_1> \
--output_jsonl_file_div <e.g._eval_for_system_prompt_modulation/values_and_system_prompt_for_principle>
```
**Required Arguments with suggestions:**
1. ``input_system_prompt_file``: Output from step 1 e.g., ``eval_for_system_prompt_modulation/values_and_system_prompt_for_principle/principle_openai.csv``

### 3. Get the values conflict and relevant dilemmas for each principle. 
```
python eval_for_system_prompt_modulation/get_values_conflicts_and_relevant_dilemmas_for_principle.py \
--input_principle_file <output_from_step_2_e.g._principle_company_clean.csv> \
--input_dilemma_file <output_from_case_1_step_2> \
--output_jsonl_file_div <eval_for_system_prompt_modulation/values_and_system_prompt_for_principle> \
--model <model_name> \
```
**Required Arguments with suggestions:**
1. ``input_principle_file``: Output from step 2. e.g., ``eval_for_system_prompt_modulation/values_and_system_prompt_for_principle/<model_name>_clean.csv``
2. ``input_dilemma_file``: Output from Step 2 of Case 1. We also provided the six models responses: ```eval/model_response_from_eval/all_models_eval_on_dilemmas_with_detail_by_action.csv```

## Case 3: Evaluate the model's steerability by system prompt modulation on each given principle
### 1. Generate system prompts for each principle -- one for steering to supporting values and one for steering to opposing values for each principle
```
python eval_for_system_prompt_modulation generate_system_prompts_for_principle.py \ 
--input_system_prompt_file <output_from_step_3_in_case_2> \
--output_jsonl_file_div <e.g._eval_for_system_prompt_modulation/values_and_system_prompt_for_principle> \
--model gpt-4-turbo \
--api_key <optional_you_can_set_as_.env_or_global>
```
**Required Arguments with suggestions:**
1. ``input_system_prompt_file``: Output from step 3 of case 2: e.g.,``<output_path>/principle_gpt4_eval_with_values_conflicts_and_dilemma.csv``

**Optional Arguments:**
1. ``api_key``: you can input OpenAI API key as an argument. Use a .env file or set as a global variable
### 2. Evaluate models on relevant dilemmas for each principle by each set of system prompts.
```
python eval_for_system_prompt_modulation/evaluate_model_on_system_prompt.py \
--input_system_prompt_file <output_from_step_1> \
--input_dilemma_file <data/dilemmas_with_detail_by_action.csv> \
--output_jsonl_file_div <e.g._eval_for_system_prompt_modulation/model_response_on_system_prompt/> \
--model gpt-4-turbo \
--api_key <optional_you_can_set_as_.env_or_global>
```
**Required Arguments with suggestions:**
1. ``input_system_prompt_file``: Output from step 1: e.g., ``<output_path>/<model>_with_system_prompts.csv``
2. ``input_dilemma_file``: MoralDilemmas data: ```data/dilemmas_with_detail_by_action.csv``
3. ``model``: ``gpt-4-turbo``

**Optional Arguments:**
1. ``api_key``: you can input OpenAI API key as an argument. Use a .env file or set as a global variable

<!-- - output: ``eval_for_system_prompt_modulation/model_response_on_system_prompt/{model}_eval.csv`` -->

### 3. Analyze the model responses from step 2 to evaluate the steerability of the model.
```
python analysis/analysis_for_system_prompt_modulation/analyze_model_resp_on_system_prompt.py \
--input_principle_file <output_from_step_2> \
--input_dilemma_file <output_from_case_1_step_2_or_provided_file> \
--output_div <e.g.analysis/analysis_for_system_prompt_modulation> \
--steer <sup/opp> \
--company_name <use_for_create_output_file_name_e.g._openai> \
--model <the_name_used_for_field_and_also_for_output_file_name>
```
**Required Arguments with suggestions:**
1. ``input_principle_file``: Output from step 2. e.g., ``eval_for_system_prompt_modulation/model_response_on_system_prompt/{model}_eval.csv``
2. ``input_dilemma_file``: Output from Step 1 in Case 1: ``eval/model_response_from_eval/<model_name>_eval_on_dilemmas_with_detail_by_action.csv`` or our provided six model responses: ``eval/model_response_from_eval/all_models_eval_on_dilemmas_with_detail_by_action.csv``
3. ``steer``: ``sup`` means that it analyzes the model responses when having system prompts steering to supporting values.

<!-- - output: ```{output_file_div}/principle_{company}_{model_name}_eval_with_system_prompt_system_prompt_sup.csv```
- output:  ```{output_file_div}/principle_{company}_{model_name}_eval_with_system_prompt_system_prompt_opp.csv``` -->

### 4. Create a plot from the analysis files from step 3. 
```
python analysis/analysis_for_system_prompt_modulation/create_plot.py \ 
--input_support_value_file <path/...system_prompt_sup_value.csv> \ 
--input_oppose_value_file <path/...system_prompt_sup_value.csv> \ 
--output_graph_div <analysis/analysis_for_system_prompt_modulation/> \
--replication_purpose <only_two_principles/or/full>
```
**Required Arguments with suggestions:**
1. ``input_[support/oppose]_value_file``: Output from step 3 with steer == [``sup``/``opp``]: ``principle_<companyname>_<modelname>_eval_with_system_prompt_system_prompt_[sup/opp]_value.csv``
2. ``replication_purpose``: [`only_two_principles`/`full`]. Depends on the evaluation of Step 1 in case 2
<!-- - We used the two output analysis files from step 3 -- one is from steering to supporting value; another one is from steering to opposing value.
- For sup and opp, you can find the output files ``principle_<companyname>_<modelname>_eval_with_system_prompt_system_prompt_<sup/opp>_value.csv`` from step 3 on: ``analysis/analysis_for_system_prompt_modulation/``. We also provided the complete files on: ``data/`` -->
<!-- - For replication_purpose, if you followed the pipeline and chose to evaluate only two principles in case 2 step 1, you can use `only_two_principles` here to have a better illustration of graphs. If you use our provided complete files, you could go for `full`. -->
