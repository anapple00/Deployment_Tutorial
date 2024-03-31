FROM python:3.9.7

WORKDIR /root/autodl-tmp/Deployment

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 6005

CMD ["uvicorn", "app", "--host", "0.0.0.0", "--post", "6005"]