import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import transformers
from transformers import AutoTokenizer
from utils import get_triples_from_zero_shot_model_output


SYSTEM_PROMPT_ZERO_SHOT = """Given the input sentence, identify all the triplets (subject, object and causal relation) \
. The subject and object should be phrases from the input sentence. 

The causal relation between subject and object should strictly be either "Positive" or "Negative" and nothing else. 

Each new extracted triplet i.e. subject, object and relation should start with a newline should be within \
<triple> and </triplet>. The subject should be within <subj> and </subj> tokens. The object should be \
within <obj> and </obj> tokens. The causal relation should be within <relation> and </relation> tokens. \
The format of output of each triplet should be strictly like below:

<triplet>
    <subj> </subj>
    <obj> </obj>
    <relation> </relation>
</triplet>
"""

def prepare_zero_shot_prompt_mistral(sentence):
    prompt = """<s>[INST] {}
Input Sentence : {} [/INST]
Causal Relation Triplet : 
    """.format(SYSTEM_PROMPT_ZERO_SHOT, sentence)
    return prompt


def prepare_zero_shot_prompt_llama2(sentence):
    prompt = """<s>[INST] <<SYS>> {}
<</SYS>>
Input Sentence : {} [/INST]
Causal Relation Triplet : 
    """.format(SYSTEM_PROMPT_ZERO_SHOT, sentence)
    return prompt


def prepare_zero_shot_prompt_llama3(sentence):    
    prompt = [
        {"role": "system", "content": "{}".format(SYSTEM_PROMPT_ZERO_SHOT)},
        {"role": "user", "content": "Input Sentence : {}".format(sentence)},
    ]
    return prompt


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser()

    # arguments for zero shot inference

    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="large language model name from huggingface")
    parser.add_argument("--saved-model-ckpt-path", required=True, type=str)
    parser.add_argument("--input-sentences-df-csv-file", type=str, required=True)
    parser.add_argument("--output-df-csv-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    args = parser.parse_args()

    model_name = args.model_name
    saved_model_ckpt_path = args.saved_model_ckpt_path
    input_sentences_df_csv_file = args.input_sentences_df_csv_file
    max_new_tokens = args.max_new_tokens
    output_df_csv_file = args.output_df_csv_file

    sentences_df = pd.read_csv(input_sentences_df_csv_file)
    try:
        if 'text' not in sentences_df.columns:
            raise KeyError("text column is not in the input sentences df csv file.")
    except KeyError as e:
        print(e)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if 'Llama-3' in model_name:
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    zero_shot_outputs = []
    extracted_triples = []
    for sentence in tqdm(sentences_df.text.to_list(), total=sentences_df.shape[0]):
        if 'Llama-2' in model_name:
            input_prompt = prepare_zero_shot_prompt_llama2(sentence)
            sequences = pipeline(
                input_prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id, # for llama2 and mistral
                max_new_tokens=512
            )
            model_output = sequences[0]['generated_text']
        elif 'Llama-3' in model_name:
            input_prompt = prepare_zero_shot_prompt_llama3(sentence)
            sequences = pipeline(
                input_prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=terminators, # for llama3 only
                max_new_tokens=512
            )
            model_output = sequences[0]['generated_text'][2]['content']
        else:
            input_prompt = prepare_zero_shot_prompt_mistral(sentence)
            sequences = pipeline(
                input_prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id, # for llama2 and mistral
                max_new_tokens=512
            )
            model_output = sequences[0]['generated_text']

        zero_shot_outputs.append(model_output)
        extracted_triples.append(
            get_triples_from_zero_shot_model_output(model_output)
        )


    sentences_df['zero_shot_model_outputs'] = zero_shot_outputs
    sentences_df['zero_shot_extracted_triples'] = extracted_triples

    sentences_df.to_csv(output_df_csv_file, index=False)

    print('Finish')
