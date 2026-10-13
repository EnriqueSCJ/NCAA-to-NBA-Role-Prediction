FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel \
    && pip install pandas numpy scipy scikit-learn imbalanced-learn xgboost catboost fastapi "uvicorn[standard]" pymilvus

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
