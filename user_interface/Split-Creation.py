import dash
from dash import dcc, html, ctx, dash_table, callback, ClientsideFunction
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import json
import math
import base64
from dash_selectable import DashSelectable
import io
import random
from striprtf.striprtf import rtf_to_text
from datetime import date
import itertools
from itertools import combinations


#### LABELED DATA LOAD ####

labeler1_labels = json.load(open("assets/labeler1_annotation.json", "r"))
labeler2_labels = json.load(open("assets/labeler2_annotation.json", "r"))
labeler3_labels = json.load(open("assets/labeler3_annotation.json", "r"))
labeler4_labels = json.load(open("assets/labeler4_annotation.json", "r"))
labeler5_labels = json.load(open("assets/labeler5_annotation.json", "r"))
labeler6_labels = json.load(open("assets/labeler6_annotation.json", "r"))
labeler7_labels = json.load(open("assets/labeler7_annotation.json", "r"))

############################
#### FILE PROCESSING #######
labeler_to_data = {
    'labeler1': labeler1_labels,
    'labeler2': labeler2_labels,
    'labeler3': labeler3_labels,
    'labeler4': labeler4_labels,
    'labeler5': labeler5_labels,
    'labeler6': labeler6_labels,
    'labeler7': labeler7_labels,
}

json_files = [labeler1_labels,labeler2_labels,labeler3_labels,labeler4_labels,labeler5_labels, labeler6_labels, labeler7_labels]

### NON FUNCTION FORMAT TO RUN ON START ###
# Step 1: Extract labelers from the JSON files
labelers = set()
u_texts = set()
texts = []
for data in json_files:
    #data = json.load(file)
    for entry in data:
        if entry['text'] not in u_texts:
            texts.append(entry['text'])
        u_texts.add(entry['text'])

        labelers.add(entry['meta_data']['labeler'])

# Step 2: Generate all possible unique combinations of two labelers
labeler_combinations = list(itertools.combinations(labelers, 2))
# Step 3: Create the mapping of each text to each combination of labelers


def random_mapping(labeler_combos,texts_,base=0):
    mapping = {}
    position = 1
    count = random.sample(range(len(labeler_combos) * 20), len(labeler_combos) * 20)
    ind = 0
    for text in texts_:
        for combo in labeler_combos:
            randomized_combo = list(combo)
            random.shuffle(randomized_combo)
            mapping[count[ind]+base] = [text, randomized_combo, position]
            ind += 1
        position += 1
    mapping = dict(sorted(mapping.items()))
    return mapping


def split_dict(data_dict):
    all_labelers = [f'labeler{i+1}' for i in range(7)]
    dupe_dict = data_dict.copy()
    split_dict = {f'labeler{i+1}': {} for i in range(7)}
    used_ids = set()
    lim = len(dupe_dict)/(len(all_labelers)*30)
    #print(lim)
    for i in range(30):
        for labeler in all_labelers:
            i = 0
            #print(labeler)
            for key, value in dupe_dict.items():
                if i == lim:
                    break
                if labeler in value[1]:
                    continue
                else:
                    if key in used_ids:
                        continue
                    used_ids.add(key)
                    split_dict[labeler][key] = value
                    i += 1

    return split_dict
            #for name in split_dict.keys():
                #if key in split_dict[name]



# Example usage:

lengths = [0,0]
attempts = 0
while sum(lengths) != 21*20:
    mapping = random_mapping(labeler_combinations,texts)
    splits = split_dict(mapping)
    lengths = [len(s) for s in splits.values()]
    print(lengths)
#for labeler, split in splits.items():
#    print(f"Split excluding {labeler}:")
#    for key, value in split.items():
#        print(f"  {key}: {value}")
print(sum(lengths))
for labeler, split in splits.items():
    print(f"Split excluding {labeler}:")
    print(len(split))

with open("SplitsJr.json", "w") as outfile:
    json.dump(splits, outfile, indent=2)

### LLM ADDITIONS ###
llama2 = json.load(open("assets/llama2_data_for_elo.json"))
llama3 = json.load(open("assets/llama3_data_for_elo.json"))
mistral = json.load(open("assets/mistral_data_for_elo.json"))
LLM_files = [llama2, mistral,llama3]
LLM_names = set()
LLM_names.add("llama2")
LLM_names.add("mistral")
LLM_names.add("llama3")

all_names = LLM_names.copy().union(labelers)
all_combinations = list(itertools.combinations(all_names, 2))
#print(llama2_annotations)
#print(mistral_annotations)
used_ids = set()
for labeler in labelers:
    inner_labelers = labelers.copy()
    inner_labelers.remove(labeler)

    LLM_temp = LLM_names.copy()
    LLM_temp.add(labeler)
    LLM_file_temp = LLM_files.copy()
    LLM_file_temp.append(labeler_to_data[labeler])
    temp_combos = [(labeler, "llama2"), (labeler, "mistral"), (labeler, "llama3")]

    lengths = [len(s) for s in splits.values()]
    temp_map = random_mapping(temp_combos,texts,sum(lengths))
    lim = 1
    for j in range(24):
        for inner_labeler in inner_labelers:
            i = 0
            for key, value in temp_map.items():
                if i == lim:
                    break
                if key in used_ids:
                    continue
                used_ids.add(key)
                splits[inner_labeler][key] = value
                i += 1
            lengths = [len(s) for s in splits.values()]
lengths = [len(s) for s in splits.values()]
print(lengths)
LLM_combinations = list(itertools.combinations(LLM_names, 2))
LLM_mapping = random_mapping(LLM_combinations, texts, sum(lengths))
lim = 1
for j in range(24):
    for labeler in labelers:
        i = 0
        for key, value in LLM_mapping.items():
            if i == lim:
                break
            if key in used_ids:
                continue
            used_ids.add(key)
            splits[labeler][key] = value
            i += 1
        lengths = [len(s) for s in splits.values()]
lengths = [len(s) for s in splits.values()]
print(lengths)
for labeler in splits.keys():
    count = random.sample(range(len(splits[labeler])), len(splits[labeler]))
    keys = list(splits[labeler].keys())
    for i in range(len(splits[labeler].keys())):
        splits[labeler][count[i]] = splits[labeler].pop(keys[i])
    splits[labeler] = dict(sorted(splits[labeler].items()))
with open("assets/Splits.json", "w") as outfile:
    json.dump(splits, outfile, indent=2)