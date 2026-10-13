FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel \
    && pip install pandas numpy pymilvus

CMD ["python", "milvus_viewer.py", "--interactive"]
