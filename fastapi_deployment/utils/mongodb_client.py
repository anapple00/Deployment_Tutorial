import os

import jsonlines
import pandas as pd
import pymongo
from loguru import logger
from tqdm import tqdm

from config.deploy import config
from utils.arguments import args

# Create a new client and connect to the server
client = pymongo.MongoClient(config.MONGODB_URL)

DATA_PATH = os.path.join(args.root_path, "data")


def _read_and_save_json(input_file):
    with open(os.path.join(dataset_path, input_file), "r", encoding="utf-8-sig") as f:
        json_data = []
        for line in tqdm(jsonlines.Reader(f), desc='Loading json file...'):  # 只有tokens和tags两个keys
            json_data.append(line)
        for idx, line in tqdm(enumerate(json_data), total=len(json_data), desc="saving json file..."):
            d = {
                "guid": f"{dataset}_{_type}_{idx}",
                "dataset": dataset,
                "dataset_type": _type,
                "tokens": line["tokens"],
                "tags": line["tags"],
            }
            try:
                x = collection.insert_one(d)
                logger.info(f"Successfully saved {dataset}_{_type}_{idx}, saving info: {x}")
            except Exception as e:
                logger.warning(f"Failed to save {dataset}_{_type}_{idx}, error info: {e}")


def _read_and_save_csv(input_file):
    df = pd.read_csv(open(os.path.join(dataset_path, input_file)))
    for idx in tqdm(df.index, desc='Loading and saving csv file...'):
        d = {
            "guid": f"{dataset}_{_type}_{idx}",
            "dataset": dataset,
            "dataset_type": _type,
            "text": df.loc[idx]["text"],
            "label": int(df.loc[idx]["label"]),  # 需要进行类别转换，不能存numpy.int64
        }
        try:
            x = collection.insert_one(d)
            logger.info(f"Successfully saved {dataset}_{_type}_{idx}, saving info: {x}")
        except Exception as e:
            logger.warning(f"Failed to save {dataset}_{_type}_{idx}, error info: {e}")


db = client["fastapi"]
for task in os.listdir(DATA_PATH):
    collection = db[task]
    task_path = os.path.join(DATA_PATH, task)
    for dataset in os.listdir(task_path):
        dataset_path = os.path.join(task_path, dataset)
        for file in os.listdir(dataset_path):
            if any((_type := ds_type) in file for ds_type in {"train", "valid", "test"}):
                if file.endswith("json"):
                    _read_and_save_json(file)
                elif file.endswith("csv"):
                    _read_and_save_csv(file)


# Send a ping to confirm a successful connection
# try:
#     db = client["fastapi"]
#     collection = db["named_entity_recognition"]
#     for d in train:
#         x = collection.insert_one(d)
# except Exception as e:
#     print(e)

print('T')
