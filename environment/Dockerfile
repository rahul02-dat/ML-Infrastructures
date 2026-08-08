FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    pytest==8.2.2 \
    pytest-json-ctrf==0.3.0

WORKDIR /app

COPY data/model.py /app/model.py
COPY data/data.py /app/data.py
COPY data/train.py /app/train.py

RUN mkdir -p /app/logs