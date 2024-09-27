import re
import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel


# prompt formats
PROMPT_LLAMA2_MISTRAL = """<s>[INST] Given the input sentence identify all the triples of entities and corresponding causal relationship between them. \
The entities should be a phrase from the input sentence and relationship should either be 'Positive' or 'Negative'. \
Every new extracted triplet should start with <triplet> token, followed by subject phrase, object phrase and relationship, separated by <sub> and <obj> tokens.
Input Sentence: {input_sentence} [/INST]
Causal Relation Triples :
"""

PROMPT_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Given the input sentence, identify all the triplets of entities and the corresponding causal relationships between them. The entities should be phrases from the input sentence, and the relationships should either be 'Positive' or 'Negative'. Each new extracted triplet should start with the <triplet> token, followed by the subject phrase, the object phrase, and the relationship, separated by <subj> and <obj> tokens. <|eot_id|><|start_header_id|>user<|end_header_id|>

Input Sentence : {input_sentence} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

TRIPLE_EXTRACTION_REGEX = re.compile(
    r'<triplet>(.*?)<subj>(.*?)<obj>\s*(positive|negative)\s*')


def get_output_from_model(
    prompt, model, input_sentence, tokenizer,
    device="cuda:0", max_new_tokens=512,
    skip_special_tokens=True
):
    prompt = prompt.format(input_sentence=input_sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_from_model = tokenizer.decode(
        outputs[0], skip_special_tokens=skip_special_tokens
    ).strip()
    return prompt, output_from_model


def create_df_of_model_outputs(
    model_name, model, tokenizer, input_sentences,
    max_new_tokens, device="cuda:0"
):
    data = [
        ['sentence', 'input_prompt', 'model_output']
    ]
    if 'Llama-3' in model_name:
        prompt = PROMPT_LLAMA3
    else:
        prompt = PROMPT_LLAMA2_MISTRAL

    for sentence in tqdm(input_sentences):
        if 'Llama-3' in model_name:
            input_prompt, output_from_model = get_output_from_model(
                prompt, model, sentence, tokenizer,
                device=device, max_new_tokens=max_new_tokens,
                skip_special_tokens=False
            )
        else:
            input_prompt, output_from_model = get_output_from_model(
                prompt, model, sentence, tokenizer,
                device=device, max_new_tokens=max_new_tokens
            )
        data.append(
            [
                sentence, input_prompt, output_from_model
            ]
        )
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

# deprecated
# def extract_triple(text):
#     regex_string = r'<triplet>(.+)<subj>(.+)<obj>(.+)'
#     match = re.search(regex_string, text)
#     if match:
#         subj = match.group(1).strip()  # Extracting triplet and extracting subject
#         obj = match.group(2).strip()      # Extracting obj
#         rel = match.group(3).strip()     # Extracting relationship
#
#         return (subj, rel, obj)
#     else:
#         return None


def extract_triples(text):
    matches = re.findall(TRIPLE_EXTRACTION_REGEX, text)
    triples = set()
    # print(matches)
    for match in matches:
        subj = match[0].strip()  # Extracting subject
        obj = match[1].strip()   # Extracting object
        rel = match[2].strip()   # Extracting relationship
        triples.add((subj, rel, obj))

    return list(triples) if triples else [None]


def get_triples_from_model_output(
        model_output, input_output_separator='[/INST]'
):
    relation_triples_part = model_output.split(input_output_separator)[-1]

    # replacing Causal Relation Triple text from
    if 'Causal Relation Triplets :' in relation_triples_part:
        relation_triples_part = relation_triples_part.replace(
            'Causal Relation Triplets :', '')

    if 'Causal Relation Triplets:' in relation_triples_part:
        relation_triples_part = model_output.replace(
            'Causal Relation Triplets', '')

    extracted_triples = []
    for sentence in relation_triples_part.split('\n'):
        sentence = sentence.strip()
        output_triple = extract_triples(sentence)
        extracted_triples.extend(output_triple)

    extracted_triples_json = []
    for item in list(set(extracted_triples)):
        if item:
            extracted_triples_json.append(
                {
                    'src': item[0],
                    'tgt': item[2],
                    'direction': item[1]
                }
            )
    return extracted_triples_json


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser()

    # arguments for inference
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="large language model name from huggingface")
    parser.add_argument("--saved-model-ckpt-path", required=True, type=str)
    parser.add_argument("--input-sentences-df-csv-file",
                        type=str, required=True)
    parser.add_argument("--output-df-csv-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    args = parser.parse_args()

    model_name = args.model_name
    saved_model_ckpt_path = args.saved_model_ckpt_path
    input_sentences_df_csv_file = args.input_sentences_df_csv_file
    max_new_tokens = args.max_new_tokens
    output_df_csv_file = args.output_df_csv_file

    # loading the base model
    # Reload model in FP16 and merge it with LoRA weights
    print('Loading the base model...')
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    # merging with lora weights
    print('Merging the base model with trained lora weights....')
    model = PeftModel.from_pretrained(
        base_model,
        saved_model_ckpt_path
    )
    model = model.merge_and_unload()

    # loading the tokenizer
    print('Loading the tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # loading input_sentences_df_csv_file
    sentences_df = pd.read_csv(input_sentences_df_csv_file)
    try:
        if 'text' not in sentences_df.columns:
            raise KeyError(
                "text column is not in the input sentences df csv file.")
    except KeyError as e:
        print(e)
        sys.exit(1)

    # making predictions and getting outputs from the model
    print('Making predictions and getting the model output...')
    outputs_df = create_df_of_model_outputs(
        model_name, model, tokenizer, sentences_df.text.to_list(),
        max_new_tokens=max_new_tokens, device="cuda:0"
    )

    # processing the model outputs and getting the triples
    print('Extracting triples from model outputs...')
    input_output_separator = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>' if 'Llama-3' in model_name else "[/INST]"

    extracted_triples = []
    for model_output in outputs_df.model_output.to_list():
        extracted_triples.append(
            get_triples_from_model_output(
                model_output, input_output_separator=input_output_separator
            )
        )

    outputs_df['extracted_triples'] = extracted_triples
    outputs_df.to_csv(output_df_csv_file, index=False)

    time_finish = time.time()
    print('Finish')
    print('Total time taken : {} minutes'.format(
        (time_finish - time_start)/60.0))
