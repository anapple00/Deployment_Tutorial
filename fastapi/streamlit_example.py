# 1. 在pycharm里写好代码 搞定
# 2. 把代码推到GitHub上 搞定
# 3. 用streamlit cloud连接GitHub，部署程序
import os
from loguru import logger
import numpy as np
import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification

emotion_map = {1: "Positive", 0: "Negative"}
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
logger.info("loading tokenizer ready...")
model = RobertaForSequenceClassification.from_pretrained("roberta-base").to('cpu')
logger.info("loading model ready...")


st.title("任原测试专用")
st.write("This is a simple emotion classification model using Roberta-base")
st.write("Test Sentence below...")

sentence = st.text_input("Enter a sentence", "Type whatever you want here")

if st.button("Answer"):
    try:
        # Tokenization
        encoded_input = tokenizer(sentence, return_tensors='pt')
        # Prediction
        outputs = model(**encoded_input)[0]
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        pred_label_ids = np.argmax(preds, axis=0)
        result = emotion_map[pred_label_ids]
        st.write(f'The emotional of this movie is {result}')
    # [today: 2, weather: 245, is 231, great: 22, !: 9877]
    # "Today weather is great!" -> [2, 245, 231, 22, 9877]
    except Exception as e:
        st.error(f"Error answering the question: {str(e)}")
