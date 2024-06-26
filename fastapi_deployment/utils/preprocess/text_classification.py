import json
import os
from dataclasses import dataclass

import pymongo

from config.deploy import config
from utils.preprocess.base import DataProcessor, InputExample


@dataclass
class MovieCommentsProcessor(DataProcessor):
    """processor for the movie comments data set."""

    def get_examples(self, data_dir, *args):
        task, dataset_name, _type = args[0], args[1], args[2]
        results = []
        for root, dirs, files in os.walk(data_dir):
            if dataset_name not in root:
                continue
            for file in files:
                if _type in file:
                    results.extend(self._read_file(os.path.join(root, file)))
        if not results:
            # load from mongoDB
            client = pymongo.MongoClient(config.MONGODB_URL)
            db = client["fastapi"]
            collection = db[task]
            for doc in collection.find({"dataset": dataset_name, "dataset_type": _type}):  # 或者{"guid": {"$regex": _type}}
                results.append({"review": doc["text"], "label": doc["label"]})
        return self._create_examples(results, _type)

    def get_labels(self, tags_map_dir):
        """get the labels."""
        tags_2_ids, ids_2_tags = dict(), dict()
        for file in os.listdir(tags_map_dir):
            if 'label' in file:
                tags_2_ids = json.load(open(os.path.join(tags_map_dir, file)))
        if tags_2_ids:
            ids_2_tags = {v: k for k, v in tags_2_ids.items()}
        return tags_2_ids, ids_2_tags

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training/dev/test sets."""
        examples = []
        for idx, line in enumerate(lines):
            guid = f"{set_type}-{idx}"
            if set_type == 'test':
                text = line[0]
                label = 1
            else:
                text = line['review']
                label = line['label']
            examples.append(InputExample(guid=guid, text=text, label=label))

        return examples
