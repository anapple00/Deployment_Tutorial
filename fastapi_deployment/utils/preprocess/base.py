import copy
import json

import jsonlines
import pandas as pd
from tqdm import tqdm


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

    def __repr__(self):
        """在打印InputExample或者对象时显示__repr__定义的信息"""
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)  # self.__dict__: 包含InptExample对象所有属性及其值的字典
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of InputExample 's for the train set."""
        raise NotImplementedError()

    def get_labels(self, tags_map_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_file(self, file_path):
        if file_path.endswith('csv'):
            return self._read_csv(file_path)
        elif file_path.endswith(('jsonl', 'json')):
            return self._read_json(file_path)

    @staticmethod
    def _read_csv(input_file):
        df = pd.read_csv(open(input_file))
        lines = []
        for idx, row in df.iterrows():
            lines.append({'review': row[df.columns[0]], 'label': row[df.columns[1]]})
        return lines

    @staticmethod
    def _read_json(input_file):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            lines = []
            for line in tqdm(jsonlines.Reader(f), desc='loading json file...'):
                lines.append(line)
            return lines
