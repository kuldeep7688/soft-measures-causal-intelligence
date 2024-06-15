import inference
import json
import os
import pymongo

MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_PORT")
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
MONGO_AUTH = os.getenv("MONGO_AUTH")
database_name = 'database_name'
collection_name = 'collection_name'

data = []
client = pymongo.MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource={MONGO_AUTH}")
db = client[database_name]
col = db[collection_name]

for doc in col.find({}):
    data.append({'passage': doc['text']})

data = data[0:50]

answers = inference.llm("TheBloke/Llama-2-70B-chat-GPTQ", data, True, False, inference.format_prompt_llama2_chat)

with open("output/auth_assets_chat_no_fine_tuning.json", "w") as f:
    json.dump(answers, f)