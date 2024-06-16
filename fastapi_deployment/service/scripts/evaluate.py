import copy
import os

import numpy as np
import torch
from loguru import logger
from sacrebleu.metrics import BLEU
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification, DataCollatorForSeq2Seq

from service.metrics.metrics import calculate_metrics
from utils.load_dataset import load_and_cache_examples

bleu = BLEU(tokenize='zh')


def evaluate(args, model, tokenizer, prefix=""):
    results = {}
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.task_name == 'named_entity_recognition':
        collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)
    elif args.task_name == 'text_classification':
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    elif args.task_name == 'seq2seq':
        collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    else:
        raise KeyError("Unknown task name .. .")
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running evaluation {prefix}*****")
    logger.info(f"Num examples: {len(eval_dataset)}")
    logger.info(f"Batch size: {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    preds, pred_label_ids = None, None  # 为预测值
    labels, out_label_ids = None, None  # 为真实标签
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # model.eval() # 位置是不是放错了，应该放到for循环外面
        inputs = {key: value.to(args.device) for key, value in batch.items()}
        if len(inputs['input_ids']) < args.eval_batch_size:
            break
        with torch.no_grad():
            if args.task_name == 'seq2seq':
                label_tokens = inputs.pop('labels').cpu().numpy()
                generated_tokens = model.generate(**inputs, max_length=128).cpu().numpy()
            else:
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if args.task_name == 'seq2seq':
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
            if not preds:
                preds = []
            preds += [pred.strip() for pred in decoded_preds]
            if not labels:
                labels = []
            labels += [[label.strip()] for label in decoded_labels]
            bleu_score = bleu.corpus_score(preds, labels).score
            results['bleu'] = bleu_score

            logger.info(f"BLEU score: {bleu_score:>0.2f}\n")
        else:
            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=preds.ndim - 2)  # 二维的在第[0]维合并，三维的在第[1]维合并
                labels = np.append(labels, inputs['labels'].detach().cpu().numpy(), axis=preds.ndim - 2)

            eval_loss = eval_loss / nb_eval_steps
            if args.task_name == "named_entity_recognition":
                pred_label_ids = np.argmax(preds, axis=2)
                # 去掉pred_Label_ids中为-100的位置
                # 计算f1前得先拍平，因为sklearn.metrics.f1_score不支持多维(1维以上)多类别(2个类别以上)分类，此处用sum(list, [])方式
                true_pred_label_ids = sum(
                    [[pred for pred, real in zip(pred_list, label_list) if real != -100] for pred_list, label_list in
                     zip(pred_label_ids, labels)], [])
                true_out_label_ids = sum(
                    [[real for pred, real in zip(pred_list, label_list) if real != -100] for pred_list, label_list in
                     zip(pred_label_ids, labels)], [])
                pred_label_ids, out_label_ids = np.array(true_pred_label_ids), np.array(true_out_label_ids)
            elif args.task_name == "text_classification":
                pred_label_ids = np.argmax(preds, axis=1)
                out_label_ids = copy.deepcopy(labels)

            result = calculate_metrics(pred_label_ids, out_label_ids)
            results.update(result)
            print()  # 换一行
            logger.info(
                f"precision: {results['precision']}, recall: {results['recall']}, f1: {results['f1']} and accuracy: {results['acc']}")
    return results
