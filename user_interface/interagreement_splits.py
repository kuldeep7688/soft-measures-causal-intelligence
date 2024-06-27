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
import pandas as pd

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
    for entry in data:
        if entry['text'] not in u_texts:
            texts.append(entry['text'])
        u_texts.add(entry['text'])

        labelers.add(entry['meta_data']['labeler'])


def remove_labeler_entries(data, labeler):
    # Create a new dictionary to store the filtered results
    filtered_data = {}
    # Create a list to store the texts in order
    texts_in_order = []

    # Iterate over the items in the original dictionary
    for key, value in data.items():  # Sorted to maintain order
        # If the labeler is not in the list, add the entry to the new dictionary
        if labeler not in value[1]:
            filtered_data[key] = value
            texts_in_order.append(value[0])

    return filtered_data, texts_in_order


def random_mapping2(labeler_combos,texts_,base=0):
    mapping = {}
    position = 1
    ind = 0
    split_dict = {f'labeler{i + 1}': {} for i in range(7)}
    for text in texts_:
        randomized_combos = random.sample(range(len(labeler_combos)), 3)
        for randomized_combo in randomized_combos:
            combo = list(labeler_combos[randomized_combo])
            random.shuffle(combo)
            mapping[ind] = [text, randomized_combo, position]
            for key in split_dict.keys():
                split_dict[key][ind] = [text, combo, position]
            ind += 1
        position += 1
    return split_dict



lengths = [0, 0]
LLM_names = set()
LLM_names.add("llama2")
LLM_names.add("mistral")
LLM_names.add("llama3")
LLM_combinations = list(itertools.combinations(LLM_names, 2))
while sum(lengths) != 3*20*7:
    mapping = random_mapping2(LLM_combinations,texts)
    #splits = split_dict(mapping)
    lengths = [len(s) for s in mapping.values()]
    print(lengths)
    print(len(lengths))
print(sum(lengths))
for labeler, split in mapping.items():
    print(f"Split excluding {labeler}:")
    print(len(split))

with open("assets/interagreement_splits.json", "w") as outfile:
    json.dump(mapping, outfile, indent=2)
