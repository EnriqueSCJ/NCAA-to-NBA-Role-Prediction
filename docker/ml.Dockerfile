FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel \
    && pip install pandas numpy scipy scikit-learn imbalanced-learn xgboost catboost optuna shap matplotlib nbformat nbconvert ipykernel pytest fastapi pymilvus openpyxl pyarrow

CMD ["python", "-m", "src.models.train"]
