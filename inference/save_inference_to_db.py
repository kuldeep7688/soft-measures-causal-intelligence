import inference
import json
import os
import pymongo
import copy

MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_PORT")
MONGO_USER_RW = os.getenv("MONGO_USER_RW")
MONGO_PASS_RW = os.getenv("MONGO_PASS_RW")
MONGO_AUTH = os.getenv("MONGO_AUTH")
database_name = 'database_name'
collection_name = 'model_factor_eval'

with open("output/auth_assets_chat_no_fine_tuning.json", "r") as f:
    answers_llama2 = inference.convert_from_llm(json.load(f))

answers_llama2 = answers_llama2[0:50]

client = pymongo.MongoClient(f"mongodb://{MONGO_USER_RW}:{MONGO_PASS_RW}@{MONGO_HOST}:{MONGO_PORT}/?authSource={MONGO_AUTH}")
db = client[database_name]
col = db[collection_name]

for answer_llama2 in list(answers_llama2)[0:100]:
    llama2_rels = copy.deepcopy(answer_llama2['rels'])
    for x in llama2_rels:
        x['correct'] = False
    db_record = {'passage': answer_llama2['passage'], 'rels': [{'model_name': 'Ground truth', 'rels': answer_llama2['rels']}, {'model_name': 'Llama 2', 'rels': llama2_rels}]}
    col.insert_one(db_record)
   