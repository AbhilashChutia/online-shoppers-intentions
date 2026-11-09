FROM python:3.10-slim

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install --no-cache-dir -r requirements.txt

COPY ["predict.py", "model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]