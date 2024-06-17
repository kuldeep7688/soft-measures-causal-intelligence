import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import transformers
from transformers import AutoTokenizer
from utils import get_triples_from_threee_shot_model_output


SYSTEM_PROMPT_THREE_SHOT = """Given the input sentence, identify all the triplets of entities and the corresponding \
causal relationships between them. The entities should be phrases from the input sentence, and the \
relationships should either be 'Positive' or 'Negative'. Each new extracted triplet should start with the \
<triplet> token, followed by the subject phrase, the object phrase, and the relationship, separated by <subj> \
and <obj> tokens.

Don't add extra sentences.
"""


def prepare_three_shot_prompt_mistral(sentence):
    prompt = """<s>[INST] {}
    Input Sentence : the current price of local rice (sold loose) at the local market is 1850 ngn/1kg. the price is expected to rise to 2100 ngn/1kg in 6 weeks, due to the high cost of oil. [/INST]
    Causal Relation Triplets : <triplet> high cost of oil <subj> price of local rice <obj> positive
    </s>

    [INST]
    Input Sentence : Participants also believed that illiteracy and low levels of education among some of the women were barriers to seeking skilled pregnancy health care. [/INST]
    Causal Relation Triplets : <triplet> illiteracy among women <subj> access to skilled pregnancy health care <obj> negative 
    <triplet> low education among women are understaffed <subj> access to skilled pregnancy health care <obj> negative
    </s>

    [INST]
    Input Sentence : Other health sources of protein are lean meats, low-fat milk, nuts, and beans such as kidney beans. [/INST]
    Causal Relation Triplets : <triplet> nuts <subj> health sources of protein <obj> positive 
    <triplet> meats <subj> health sources of protein <obj> positive
    <triplet> milk <subj> health sources of protein <obj> positive
    <triplet> beans <subj> health sources of protein <obj> positive
    </s>

    [INST]
    Input Sentence : {} [/INST]
    Causal Relation Triplets : 
        """.format(SYSTEM_PROMPT_THREE_SHOT, sentence)
    return prompt


def prepare_three_shot_prompt_llama2(sentence):
    prompt = """<s>[INST] <<SYS>> {}
    <</SYS>>
    Input Sentence : the current price of local rice (sold loose) at the local market is 1850 ngn/1kg. the price is expected to rise to 2100 ngn/1kg in 6 weeks, due to the high cost of oil. [/INST]
    Causal Relation Triplets : <triplet> high cost of oil <subj> price of local rice <obj> positive
    </s>

    [INST]
    Input Sentence : Participants also believed that illiteracy and low levels of education among some of the women were barriers to seeking skilled pregnancy health care. [/INST]
    Causal Relation Triplets : <triplet> illiteracy among women <subj> access to skilled pregnancy health care <obj> negative 
    <triplet> low education among women are understaffed <subj> access to skilled pregnancy health care <obj> negative
    </s>

    [INST]
    Input Sentence : Other health sources of protein are lean meats, low-fat milk, nuts, and beans such as kidney beans. [/INST]
    Causal Relation Triplets : <triplet> nuts <subj> health sources of protein <obj> positive 
    <triplet> meats <subj> health sources of protein <obj> positive
    <triplet> milk <subj> health sources of protein <obj> positive 
    <triplet> beans <subj> health sources of protein <obj> positive 
    </s>

    [INST]
    Input Sentence : {} [/INST]
    Causal Relation Triplets : 
        """.format(SYSTEM_PROMPT_THREE_SHOT, sentence)
    
    return prompt


def prepare_three_shot_prompt_llama3(sentence):    
    prompt = [
        {"role": "system", "content": "{}".format(SYSTEM_PROMPT_THREE_SHOT)},
        {"role": "user", "content": "Input Sentence : the current price of local rice (sold loose) at the local market is 1850 ngn/1kg. the price is expected to rise to 2100 ngn/1kg in 6 weeks, due to the high cost of oil."},
        {"role": "assistant", "content": "Causal Relation Triplets : <triplet> high cost of oil <subj> price of local rice <obj> positive"},
        {"role": "user", "content": "Input Sentence : Participants also believed that illiteracy and low levels of education among some of the women were barriers to seeking skilled pregnancy health care."},
        {"role": "assistant", "content": "Causal Relation Triplets : <triplet> illiteracy among women <subj> access to skilled pregnancy health care <obj> negative "},
        {"role": "user", "content": "Input Sentence : Other health sources of protein are lean meats, low-fat milk, nuts, and beans such as kidney beans."},
        {"role": "assistant", "content": "Causal Relation Triplets : <triplet> nuts <subj> health sources of protein <obj> positive \n <triplet> meats <subj> health sources of protein <obj> positive \n <triplet> milk <subj> health sources of protein <obj> positive \n <triplet> beans <subj> health sources of protein <obj> positive"},
        {"role": "user", "content": "Input Sentence : {}".format(sentence)},
    ]
    
    return prompt


if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser()

    # arguments for three shot inference
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
            input_prompt = prepare_three_shot_prompt_llama2(sentence)
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
            input_prompt = prepare_three_shot_prompt_llama3(sentence)
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
            input_prompt = prepare_three_shot_prompt_mistral(sentence)
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
            get_triples_from_threee_shot_model_output(model_output)
        )

    sentences_df['three_shot_model_outputs'] = zero_shot_outputs
    sentences_df['three_shot_extracted_triples'] = extracted_triples

    sentences_df.to_csv(output_df_csv_file, index=False)
    time_stop = time.time()
    print('Finish')
    print('Total time taken = {} minutes'.format((time_stop - time_start)/60.0))
