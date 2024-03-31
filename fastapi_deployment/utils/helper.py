from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from loguru import logger


def get_optimizer_and_scheduler(model, args, total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    return optimizer, scheduler


def decode_ner_output(input_ids, pred_label_ids, id2label, tokenizer):
    # answer: [0,0,0,0,0,0,0,0,0,3,4,0,3,0,0,5,6,6,0,0,0,0,0,0,0,0]
    answer_dict = dict()
    answer_cache = []  # 用来存已经制作好的words
    entity_type = ""
    sub_token_list = []  # 用来存已经制作好的sub-tokens
    token_list = tokenizer.convert_ids_to_tokens(input_ids)
    logger.info(f"Input tokens: {token_list}")
    logger.info(f"Predict labels: {pred_label_ids}")
    for idx, (token, label) in enumerate(zip(token_list, pred_label_ids)):
        if id2label[label].startswith('O'):
            if answer_cache and entity_type:
                answer_dict[" ".join(answer_cache)] = entity_type
                answer_cache, entity_type = [], ""
        if (entity := id2label[label]).startswith('B') and token.startswith('Ġ'):
            sub_token_list = [token.strip('Ġ')]
            entity_type = entity.split('-', maxsplit=1)[-1]
            next_idx = idx + 1
            while not token_list[next_idx].startswith('Ġ') and pred_label_ids[next_idx] == label:
                sub_token_list.append(token_list[next_idx])
                next_idx += 1
        elif id2label[label].startswith('I') and token.startswith('Ġ'):
            sub_token_list = [token.strip('Ġ')]
            next_idx = idx + 1
            while not token_list[next_idx].startswith('Ġ') and pred_label_ids[next_idx] == label and \
                    id2label[pred_label_ids[next_idx]].split('-', maxsplit=1)[-1] == entity_type:
                sub_token_list.append(token_list[next_idx])
                next_idx += 1
        # 每次检查一下
        if sub_token_list:
            answer_cache.append(''.join(sub_token_list))
            sub_token_list = []
    return answer_dict
