"""
0. 实例化任务所需的tokenizer和model
1. 解析swagger接收的InputSchema格式的数据
2. 把第1步解析的数据加载InputExample类，再对其中的文本做tokenization并用InputFeatures类接收
3. 用CustomDataset类接收第2步得到的InputFeatures类数据
4. 实例化DataCollator，用于给CustomDataset中的数据加padding
5. 把第3步得到的CustomDataset放到DotaLoader里，并把第4步得到的DataCollator作为参数传入
6，用for循环训练DataLoader中的数据(只有1条)，
7. 用模型预测结果
8. model中带的id2label字典解析预测的结果
"""
import numpy as np
import torch
from loguru import logger
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification, DataCollatorForSeq2Seq

from utils.helper import decode_ner_output
from utils.load_dataset import load_and_cache_examples


def predict(args, model, tokenizer, prefix=""):
    pred_dataset = load_and_cache_examples(args, tokenizer, predict=True)

    if args.task_name == 'named_entity_recognition':
        collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)
    elif args.task_name == 'text_classification':
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    elif args.task_name == 'seq2seq':
        collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    else:
        raise KeyError("Unknown task name ...")

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset,
                                 sampler=pred_sampler,
                                 batch_size=1,
                                 collate_fn=collate_fn)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Predict!
    logger.info(f"***** Running {prefix} *****")
    logger.info(f"Num examples {len(pred_dataset)}")
    logger.info(f"Batch size {args.eval_batch_size}")
    pred_label_ids = None
    response = ""  # 构造的AI字符串形式输出
    for batch in tqdm(pred_dataloader, desc="predicting"):
        # model.eval()
        inputs = {key: value.to(args.device) for key, value in batch.items()}

        with torch.no_grad():
            if args.task_name == 'seq2seq':
                generated_tokens = model.generate(**inputs, max_length=128).cpu().numpy()
            else:
                outputs = model(**inputs)
                logits = outputs.logits
                preds = logits.detach().cpu().numpy()
                pred_label_ids = np.argmax(preds, axis=2) if len(logits.size()) == 3 else np.argmax(preds, axis=1)

        if args.task_name == 'seq2seq':
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            result = decoded_preds[0]
            response = f"The translation: {result}"
        elif args.task_name == 'text_classification':
            pred_label_ids = pred_label_ids.tolist()[0]
            result = model.config.id2label.get(pred_label_ids, "")
            response = f"The characteristic is: {result}"
        elif args.task_name == 'named_entity_recognition':
            input_ids = inputs['input_ids'].tolist()
            pred_label_ids = pred_label_ids.tolist()
            result = decode_ner_output(input_ids[0], pred_label_ids[0], model.config.id2label, tokenizer)
            response = "The entity with category: " + '; '.join([f'"{key}": {value}' for key, value in result.items()])
    return response
