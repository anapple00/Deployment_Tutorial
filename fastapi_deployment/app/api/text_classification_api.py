import os

import torch
from fastapi import APIRouter
from loguru import logger

from fastapi_deployment.common.schema.class_mapping import MODEL_CLASSES
from fastapi_deployment.common.schema.input_output import InputSchema, OutputSchema
from fastapi_deployment.service.scripts.inference import predict
from fastapi_deployment.utils.arguments import args

router = APIRouter(tags=["Available AI Models"])


@router.post("/text classification", response_model=OutputSchema)
def run_text_classification_service(data: InputSchema):
    args.query = data.query
    args.model_type = data.model_type
    args.task_name = data.task
    args.dataset_name = data.dataset
    logger.debug(f"[Start Task]: {''.join(args.task_name.split(' ')).title()}")
    logger.info(f"Input query: {args.query}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    save_path = os.path.join(args.output_dir, args.task_name)
    tokenizer = tokenizer_class.from_pretrained(save_path,
                                                do_lower_case=args.do_lower_case)
    logger.info(f"Predicting the following checkpoints: {save_path}")

    model = model_class.from_pretrained(save_path)
    model.to(args.device)

    prediction = predict(args, model, tokenizer, prefix="predict")
    result_schema = OutputSchema(response=prediction)

    return result_schema
