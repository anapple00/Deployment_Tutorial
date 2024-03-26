"""
0. 实例化任务所需的tokenizer和model
1. 读取数据(json或csv格式)，每条数据用InputExample类接收
2，针对InputExample中的文本做tokenization，每条数据用InputFeatures类接收
3. 用CustomDataset类收集所有InputFeatures类数据
4. 实例化DataCollator，用于给CustomDataset中的数据加padding
5. 把第3步得到的CustomDataset放到DataLoader里，并把第4步得到的DataCollator作为参数传入
6. 加载optimizer和scheduler
7. 用for循环训练DataLoader中的数据，每次训练一个batch的数据
8. 每当经过step步的时候用evaluate函数评估模型，若此时F1更好则保存模型(不保存tokenizer)
9. 保存完整模型(tokenizer+model)
"""
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import RandomSampler, DataLoader, DistributedSampler
from tqdm import trange, tqdm
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification, DataCollatorForSeq2Seq

from fastapi.common.schema.class_mapping import MODEL_CLASSES, DATASET_TYPES, TASK_TYPES
from fastapi.service.scripts.evaluate import evaluate
from fastapi.utils.arguments import args
from fastapi.utils.helper import get_optimizer_and_scheduler
from fastapi.utils.load_dataset import load_and_cache_examples


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.task_name == 'named_entity_recognition':
        collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)
    elif args.task_name == 'text_classification':
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    elif args.task_name == 'seq2seq':
        collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    else:
        raise KeyError("Unknown task name...")
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer, scheduler = get_optimizer_and_scheduler(model, args, t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples: {len(train_dataset)}")
    logger.info(f"Num Epochs: {args.num_train_epochs}")
    logger.info(f"Instantaneous batch size per GPU: {args.per_gpu_train_batch_size}")
    logger.info(
        f"Total train batch size (w. parallel, distributed & accumulation): {args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)}")
    logger.info(f"Gradient Accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {t_total}")

    max_score, metrics_score_list = 0.0, []
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {kev: value.to(args.device) for kev, value in batch.items()}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                # if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps  # 每batch都将Loss除以gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            epoch_iterator.set_description("loss {}".format(round(loss.item(), 5)))

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:  # 过gradient_accumulation_steps后才将梯度清零
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # 每Logging_steps，进行evaluate
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        metrics_score_list.append(results.get('f1', results.get('bleu', 0.0)))
                        for key, value in results.items():
                            eval_key = f'eval_{key}'
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                # 每save steps保存checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if not max(metrics_score_list) > max_score:
                        continue
                    max_score = max(metrics_score_list)
                    # Save model checkpoint
                    output_dir = Path(args.output_dir) / args.task_name / ('_'.join(
                        [args.model_name_or_path.split('/')[-1], args.dataset_name,
                         f'checkpoint-{format(global_step)}']))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.maxsteps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


if __name__ == "__main__":

    # 是否覆盖输出目录
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "output directory ([}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/6PUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning(
        f"process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}, distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}")

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in TASK_TYPES or args.dataset_name not in DATASET_TYPES:
        raise ValueError(f"Task not found: {args.task_name} or Dataset not found: {args.dataset_name}")

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type.lower()]
    processor = DATASET_TYPES[args.dataset_name]()
    tag_2_ids, ids_2_tags = processor.get_labels(Path(args.data_dir) / args.task_name / args.dataset_name)
    num_labels = len(tag_2_ids.keys())
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          id2label=ids_2_tags,
                                          label2id=tag_2_ids,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                        ignore_mismatched_sizes=True)

    # print model config
    logger.info(f"Model config {str(config)}")

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    #  log输出训练/评估超参数
    logger.info("==== Training/Evaluation Parameters: =====")
    for attr, value in sorted(args.__dict__.items()):
        logger.info(f'\t{attr}={value}')
    logger.info("==== Parameters End =====\n")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(f"global_step: global_step), average loss: {tr_loss}")

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        save_path = os.path.join(args.output_dir, args.task_name)
        if not os.path.exists(save_path) and args.local_rank in [-1, 0]:
            os.makedirs(save_path)
        logger.info(f"Saving model checkpoint to {save_path}")
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`.
        model_to_save = model.module if hasattr(model,
                                                ' module') else model  # Take care of distributed/parallel
        model_to_save.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(save_path, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(save_path)
        tokenizer = tokenizer_class.from_pretrained(save_path)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        save_path = os.path.join(args.output_dir, args.task_name)
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.output_dir, args.task_name),
                                                    do_lower_case=args.do_lower_case)
        checkpoints = [save_path]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(save_path + '/**/' + 'model.safetensors', recursive=True)))
            logger.info(f"Evaluate the following checkpoints: {checkpoints}")
        for checkpoint in checkpoints:
            globalstep = checkpoint.split(' -')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args, device)

            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(globalstep), v) for k, v in result.items())
            results.update(result)
