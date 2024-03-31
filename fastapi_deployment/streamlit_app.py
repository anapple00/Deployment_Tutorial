import asyncio
import random

import streamlit as st
from loguru import logger

from app.api import run_ner_service
from common.schema.class_mapping import MODEL_CLASSES, TASK_TYPES, DATASET_TYPES
from common.schema.input_output import InputSchema


def main():
    st.set_page_config(page_title="AI Model Deployment", page_icon=":shark:", layout="wide")  # 会展示在浏览器标签页上
    st.title('AI Model Deployment:robot_face:')
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
    else:
        st.session_state.first_visit = False

    # 初始化全局配置
    if st.session_state.first_visit:
        st.session_state.random_model_index = random.choice(range(len(MODEL_CLASSES)))
        st.session_state.random_task_index = random.choice(range(len(TASK_TYPES)))
        st.session_state.random_dataset_index = random.choice(range(len(DATASET_TYPES)))

    st.sidebar.write(f'Choose basic configuration')
    model_type = st.sidebar.selectbox('Select the AI model', MODEL_CLASSES.keys(),
                                      index=st.session_state.random_model_index)
    task_name = st.sidebar.selectbox('select the task type', TASK_TYPES, index=st.session_state.random_task_index)
    dataset_name = st.sidebar.selectbox('Select the dataset type', DATASET_TYPES.keys(),
                                        index=st.session_state.random_dataset_index)
    st.markdown(f'### You Would Like to do {task_name.replace("_", " ").title()} Task')
    secret = st.text_input('Please enter the test query',
                           "The study demonstrated a decreased level of glucocorticoid receptors (GR) in")

    if st.button("Answer"):
        data = InputSchema(query=secret, model_type=model_type, task=task_name, dataset=dataset_name)
        logger.info("Input data is ready.")
        result = asyncio.run(run_ner_service(data))
        st.write("Result:", result)


if __name__ == "__main__":
    main()
