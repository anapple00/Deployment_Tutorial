FROM python:3.10.14

WORKDIR .

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./fastapi_deployment ./fastapi_deployment

CMD ["python", "fastapi_deployment/main.py"]
