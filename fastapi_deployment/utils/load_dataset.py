import os
import pickle
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from fastapi.common.schema.class_mapping import DATASET_TYPES
from fastapi.common.schema.input_features import InputFeatures
from fastapi.utils.dataset import CustomDataset
from fastapi.utils.preprocess.base import InputExample


def load_and_cache_examples(args, tokenizer, evaluate=False, predict=False) -> CustomDataset:
    """
    将dataset转换为features，并保存在目录cached_features_file中
    args:
        evaluate: False。为True，则对dev.csv进行转换
        predict: False。若为True，则则对test.csv进行转换
    return:
        dataset
    """
    task = args.task_name
    dataset_name = args.dataset_name
    if args.local_rank not in [-1, ] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process th

    processor = DATASET_TYPES[dataset_name]()

    # Load data features from cache or dataset file
    if evaluate:
        exec_model = 'dev'
    elif predict:
        exec_model = 'test'
    else:
        exec_model = 'train'

    if exec_model == 'test':
        examples = []
        guid = f"{exec_model}-{0}"
        text = args.query
        label = ''
        examples.append(InputExample(guid=guid, text=text, label=label))
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                max_length=tokenizer.model_max_length,
                                                )
        dataset = CustomDataset(features)
        return dataset

    cached_file_dir = Path(args.data_dir) / task / dataset_name
    cached_file_path = cached_file_dir / ('_'.join(['cached', exec_model, dataset_name]) + '.pkl')
    if os.path.exists(cached_file_path) and not args.overwrite_cache:
        logger.info(f"Loading features from cached file {cached_file_path}")
        features = pickle.load(open(cached_file_path, 'rb'))
    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")
        if evaluate:
            examples = processor.get_examples(args.data_dir, dataset_name, 'valid')
        else:
            examples = processor.get_examples(args.data_dir, dataset_name, 'train')
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                max_length=tokenizer.model_max_length
                                                )
        if args.local_rank in [-1, 0]:
            logger.info(f"Saving features into cached file {cached_file_path}")
            pickle.dump(features, open(cached_file_path, 'wb'))

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Build dataset
    dataset = CustomDataset(features)
    return dataset


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_length=512,
                                 ) -> list[InputFeatures]:
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method

    Returns:
        If the * examples ** input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    features = []
    for ex_index, example in tqdm(enumerate(examples), desc="Encoding and padding process...", total=len(examples)):
        if ex_index and ex_index % 10000 == 0:
            logger.info(f"Writing example {ex_index}")

        if not example.label or isinstance(example.label, int):  # 针对text classification任务或预测任务
            inputs = tokenizer.encode_plus(
                example.text,
                add_special_tokens=True,
                max_length=max_length,
            )

            input_ids = inputs["input_ids"]
            real_token_len = len(input_ids)
            attention_mask = inputs["attention_mask"]
            labels = example.label if isinstance(example.label, int) else None

        elif isinstance(example.label, str):  # 针对sequence to sequence任务
            input_encoded = tokenizer(
                example.text,
                padding=True,
                max_length=max_length,
                truncation=True,
            )
            input_ids = input_encoded["input_ids"]
            attention_mask = input_encoded["attention_mask"]
            real_token_len = len(input_ids)
            with tokenizer.as_target_tokenizer():
                target_encoded = tokenizer(
                    example.label,
                    padding=True,
                    max_length=max_length,
                    truncation=True,
                )
                labels = target_encoded["input_ids"]
        else:  # 针对NER任务
            input_ids, attention_mask, labels = [tokenizer.bos_token_id], [0], [0]
            for token, token_label in zip(example.text, example.label):
                input_ids.extend(tokenizer(token, add_special_tokens=False)['input_ids'])  # 切sub-tokens
                attention_mask.extend(tokenizer(token, add_special_tokens=False)['attention_mask'])
                labels.extend([token_label] * len(tokenizer(token, add_special_tokens=False)['input_ids']))
            real_token_len = len(input_ids)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"guid:{example.guid}")
            logger.info(f"real_token_len: {real_token_len}")
            logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            logger.info(f"attention_mask: {' '.join([str(x) for x in attention_mask])}")

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          label=labels,
                          real_token_len=real_token_len))

    return features
