import os

import boto3
import torch
from fastapi import APIRouter
from loguru import logger

from common.schema.class_mapping import MODEL_CLASSES
from common.schema.input_output import InputSchema, OutputSchema
from service.scripts.inference import predict
from utils.arguments import args

router = APIRouter(tags=["Available AI Models"])


@router.post("/ner", response_model=OutputSchema)
async def run_ner_service(data: InputSchema):
    args.query = data.query
    args.model_type = data.model_type
    args.task_name = data.task
    args.dataset_name = data.dataset
    logger.debug(f"[Start Task]: {''.join(args.task_name.split('_')).title()}")
    logger.info(f"Input query: {args.query}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.local:
        save_path = os.path.join(args.output_dir, args.task_name)
        tokenizer = tokenizer_class.from_pretrained(save_path,
                                                    do_lower_case=args.do_lower_case)
        logger.info(f"Predicting the following checkpoints: {save_path}")
        model = model_class.from_pretrained(save_path)
    else:
        from aws_secrets import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        args.aws_id, args.aws_key = AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        s3 = boto3.client(
            's3',
            aws_access_key_id=args.aws_id,
            aws_secret_access_key=args.aws_key,
        )
        tokenizer_config = s3.get_object(Bucket=args.aws_bucket, Key=f"fastapi/{args.task_name}/tokenizer_config.json")[
            'Body'].read().decode('utf-8')
        logger.debug(f"Got tokenizer cache in AWS s3")
        config = s3.get_object(Bucket=args.aws_bucket, Key=f"fastapi/{args.task_name}/config.json")[
            'Body'].read().decode('utf-8')
        logger.debug(f"Got config cache in AWS s3")
        model_checkpoint = s3.get_object(Bucket=args.aws_bucket, Key=f"fastapi/{args.task_name}/model.safetensors")[
            'Body'].read().decode('utf-8')
        logger.debug(f"Got model checkpoints in AWS s3")

        tokenizer = tokenizer_class.from_pretrained(tokenizer_config,
                                                    do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(model_checkpoint, config=config)

    model.to(args.device)

    prediction = predict(args, model, tokenizer, prefix="predict")
    result_schema = OutputSchema(response=prediction)

    return result_schema
