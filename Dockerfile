FROM python:3.10

WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY anomaly_model.py .
COPY model.pth .
COPY data.csv .
COPY static /app/static
COPY templates /app/templates

RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
