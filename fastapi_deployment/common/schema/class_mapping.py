from transformers import (
    BertConfig, BertForSequenceClassification, BertTokenizer,
    RobertaConfig, RobertaForSequenceClassification, RobertaForTokenClassification, RobertaTokenizer,
    AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
)

from utils.preprocess import MovieCommentsProcessor, BiologyProcessor, TranslationProcessor

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'roberta_ner': (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    'auto': (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer)
}

TASK_TYPES = {'named_entity_recognition', 'text_classification', 'seq2seq'}

DATASET_TYPES = {
    "movie_comments": MovieCommentsProcessor,
    "biology": BiologyProcessor,
    "english_chinese_translation": TranslationProcessor
}
