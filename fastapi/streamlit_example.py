# 1. 在pycharm里写好代码 搞定
# 2. 把代码推到GitHub上 搞定
# 3. 用streamlit cloud连接GitHub，部署程序
import os

import numpy as np
import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification

os.environ['HTTPS_PROXY'] = "https://orangelgy_qifei:123tiancai@global-jp.link.ac.cn:443"
emotion_map = {1: "Positive", 0: "Negtive"}
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base")

st.title("任原测试专用")
st.write("This is a simple emotion classification model using Roberta-base")
st.write("Test Sentence below...")

sentence = st.text_input("Enter a sentence", "Type whatever you want here")

if st.button("Answer"):
    try:
        # tokenize
        encoded_input = tokenizer(sentence, return_tensors='pt')
        # predict
        outputs = model(**encoded_input)[0]
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        pred_label_ids = np.argmax(preds, axis=0)
        result = emotion_map[pred_label_ids]
    # [today: 2, weather: 245, is 231, great: 22, !: 9877]
    # "Today weather is great!" -> [2, 245, 231, 22, 9877]
    except Exception as e:
        st.error(f"Error answering the question: {str(e)}")
