import requests
import json
import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

OUTPUT_DIR_PREFIX = "fine_tune_output"
FINAL_CHECKPOINT_SUFFIX = "final_checkpoints"
CACHE_MODELS = {}
CACHE_TOK = {}
PROMPT_PREFIX = "You will be presented with a passage. Extract from it pairs of concepts for which a change in the magnitude of one concept would (putatively) causally change the magnitude of the second concept. Only include concepts explicitly mentioned and only include direct relationships. Do not include indirect or implied relationships. Report the extracted concepts and the ways in which their magnitudes covary only; keep the answer as brief as possible to capture this information. Do not explain the answer, just state it.{}\n\nPassage:\n\n{}\n\nRelationships\n\n"

def prompt_and_response_train(datum, prompt_format_func):
    prompt = f"Passage:\n\n{datum['passage']}\n\nRelationships"
    response = ""
    for i,rel in enumerate(datum['rels']):
        response += f"\n\nRelationship {i+1}: a {rel['direction_head']} in the magnitude of [{rel['head']}] results in a {rel['direction_tail']} in the magnitude of [{rel['tail']}]"
    if len(datum['rels']) == 0:
        response += "\n\n<|NO RELATIONSHIPS FOUND|>"
    return prompt_format_func(prompt, response)

def convert_from_llm(llm_output):
    results = []
    for answer in llm_output:
        passage = re.match('Passage:\s+(.*?)\n\nRelationships\n\n', answer, re.DOTALL).group(1)
        factor = {'passage': passage, 'rels': []}
        for m in re.finditer('Relationship (\d+): an? (.+?) in the .+? of \[(.+?)\] results in an? (.+?) in the .+? of \[(.+?)\]', answer):
            rel = {'head': m.group(3), 'tail': m.group(5), 'direction_head': fix_rel_direction(m.group(2)), 'direction_tail': fix_rel_direction(m.group(4))}
            if not has_equivalent_rel(factor['rels'], rel):
                factor['rels'].append(rel)
        results.append(factor)
    return results

def fix_rel_direction(direction):
    if re.match("increases?", direction, re.IGNORECASE):
        return "increase"
    elif re.match("decr?eases?", direction, re.IGNORECASE):
        return "decrease"
    else:
        return "Can't parse direction"

def has_equivalent_rel(data, rel):
    for x in data:
        if x['head'] == rel['head'] and x['tail'] == rel['tail'] and \
           ((x['direction_head'] == "increase" and rel['direction_head'] == "decrease") or (x['direction_head'] == "decrease" and rel['direction_head'] == "increase")) and \
           ((x['direction_tail'] == "increase" and rel['direction_tail'] == "decrease") or (x['direction_tail'] == "decrease" and rel['direction_tail'] == "increase")):
            return True
    return False

def load_model(model_name, is_fine_tuned, orig_model_name=None):
    if model_name not in CACHE_MODELS:
        path = model_name if not is_fine_tuned else os.path.join(OUTPUT_DIR_PREFIX, model_name, FINAL_CHECKPOINT_SUFFIX)
        model_loader = AutoModelForCausalLM if not is_fine_tuned else AutoPeftModelForCausalLM
        CACHE_MODELS[model_name] = model_loader.from_pretrained(path, device_map="auto")
        CACHE_TOK[model_name] = AutoTokenizer.from_pretrained(model_name if orig_model_name is None else orig_model_name, use_fast=True)
    model = CACHE_MODELS[model_name]
    tokenizer = CACHE_TOK[model_name]
    return model, tokenizer

def format_prompt_llama2_chat(prompt, response=None):
    #preamble = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    preamble = "You are a computer program which extracts pairs of concepts whose magnitudes covary from passages of text. You do not have a personality, you only output the information requested without extra conversation."
    response = response + " </s>" if response is not None else ""
    return f"[INST] <<SYS>>\n{preamble}\n<</SYS>>\n\n{prompt} [/INST] {response}"

def format_prompt_llama2_base(prompt, response=None):
    response = response + " </s>" if response is not None else ""
    return f"{prompt}{response}"

def llm(model_name, data, include_examples, is_fine_tuned, prompt_format_func, orig_model_name=None):
    model, tokenizer = load_model(model_name, is_fine_tuned, orig_model_name)
    examples = "Here is an example:\n\n"
    examples += NSHOT_EXAMPLES
    examples += "\n\nNow it's your turn."
    answers = []
    for i,x in enumerate(data):
        print(f"{i}/{len(data)-1}")
        prompt = prompt_format_func(PROMPT_PREFIX.format(examples if include_examples else "", x['passage']))
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        answer = tokenizer.decode(model.generate(inputs=input_ids, penalty_alpha=0.6, top_k=4, max_new_tokens=512)[0])
        *_, last_passage = re.finditer("Passage:", answer)
        answer = answer[last_passage.start():]
        answers.append(answer)
    return answers

NSHOT_EXAMPLES = """Passage:

Living in a bustling city, it's no surprise that the more people there are crammed into a limited space, the more chaotic the roads become. This results in more accidents and ultimately higher insurance rates. Population density also increases the demand for common services such as child care, while scarcity of space and other factors put constraints on supply. The overall result is increased costs across the board and less money left over from one's paycheck.

Relationships

Relationship 1: a increase in the magnitude of [urban population density] results in a increase in the magnitude of [chaotic roads]

Relationship 2: a increase in the magnitude of [accidents] results in a increase in the magnitude of [insurance rates]

Relationship 3: a increase in the magnitude of [urban population density] results in a increase in the magnitude of [demand for common services such as child care]

Relationship 4: a decrease in the magnitude of [living space] results in a decrease in the magnitude of [supply of common services]

Relationship 5: a increase in the magnitude of [insurance rates] results in a increase in the magnitude of [overall costs]

Relationship 6: a increase in the magnitude of [demand for common services such as child care] results in a increase in the magnitude of [overall costs]

Relationship 7: a decrease in the magnitude of [supply of common services] results in a increase in the magnitude of [overall costs]

Relationship 8: a increase in the magnitude of [insurance rates] results in a decrease in the magnitude of [money left over from one's paycheck]

Relationship 9: a increase in the magnitude of [demand for common services such as child care] results in a decrease in the magnitude of [money left over from one's paycheck]

Relationship 10: a increase in the magnitude of [chaotic roads] results in a increase in the magnitude of [accidents]

Relationship 11: a decrease in the magnitude of [supply of common services] results in a decrease in the magnitude of [money left over from one's paycheck]

Here is a second example:

Example 2

Passage:

People generally connect to the internet through their smartphones or laptops. At work, people also use workstations.

Relationship 1: a increase in the magnitude of [number of smartphones] results in a increase in the magnitude of [access to the internet]

Relationship 2: a increase in the magnitude of [number of laptops] results in a increase in the magnitude of [access to the internet]

Relationship 3: a increase in the magnitude of [number of workstations at work] results in a increase in the magnitude of [access to the internet]

Here is a final example of another passage, where this time there are no concepts whose magnitudes covary.

Example 3

Passage:

Urban population density refers to the measurement of how many people reside within a given area of an urban environment. It is typically calculated by dividing the total population of a city or town by its land area. This concept provides insights into the concentration of individuals in urban areas and helps policymakers and urban planners understand the intensity of human activity within a specific space. Urban population density can vary greatly across different cities, reflecting diverse patterns of urbanization and development.

Relationships

<|NO RELATIONSHIPS FOUND|>"""