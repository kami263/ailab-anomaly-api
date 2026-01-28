import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from anomaly_model import (
    AutoEncoder,
    train_autoencoder_from_csv,
    load_model_and_threshold
)

# ---------------------------------------------------------
# JSON ログフォーマッタ
# ---------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        return json.dumps(log)

# uvicorn ロガーに適用
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers:
        handler.setFormatter(JsonFormatter())

# ---------------------------------------------------------
# FastAPI アプリ
# ---------------------------------------------------------
app = FastAPI(
    title="Anomaly Detection API",
    description="AutoEncoder を用いた異常検知 API",
    version="1.0.0"
)

# 静的ファイル
app.mount("/static", StaticFiles(directory="static"), name="static")

# Swagger UI カスタム
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    try:
        with open("templates/custom_swagger.html") as f:
            html = f.read()
        return HTMLResponse(html)
    except FileNotFoundError:
        from fastapi.openapi.docs import get_swagger_ui_html
        return get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")

# ---------------------------------------------------------
# 共通関数
# ---------------------------------------------------------
def now_jst():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()

# ---------------------------------------------------------
# Response Models
# ---------------------------------------------------------
class AnomalyResponse(BaseModel):
    anomaly_score: float
    threshold: float
    status: str

class PredictResponse(BaseModel):
    anomaly_score: float
    threshold: float
    status: str
    model_version: str | None = None
    timestamp: str | None = None
    input: Dict[str, float] | None = None

# ---------------------------------------------------------
# モデルロード（startup イベントで実行）
# ---------------------------------------------------------
CSV_PATH = "/app/data.csv"
MODEL_PATH = "/app/model.pth"

@app.on_event("startup")
def load_models():
    """
    アプリ起動時にモデルをロードする
    """
    logging.info("Loading models...")

    # /anomaly 用モデル
    df = pd.read_csv(CSV_PATH)
    input_dim = df.select_dtypes(include=["number"]).shape[1]
    app.state.input_dim = input_dim

    if os.path.exists(MODEL_PATH):
        model = AutoEncoder()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        app.state.model_anomaly = model
        app.state.threshold_anomaly = 1.0  # 必要なら外部化
        logging.info("Loaded saved anomaly model.")
    else:
        model, mean, std, threshold = train_autoencoder_from_csv(CSV_PATH)
        app.state.model_anomaly = model
        app.state.mean_anomaly = mean
        app.state.std_anomaly = std
        app.state.threshold_anomaly = threshold
        logging.info("Trained new anomaly model.")

    # /predict 用モデル
    model, mean, std, threshold = load_model_and_threshold()
    app.state.model_predict = model
    app.state.mean_predict = mean
    app.state.std_predict = std
    app.state.threshold_predict = threshold

    logging.info("All models loaded successfully.")

# ---------------------------------------------------------
# /anomaly
# ---------------------------------------------------------
class InputData(BaseModel):
    values: List[float]

@app.post(
    "/anomaly",
    tags=["Anomaly Detection"],
    summary="Detect anomaly from numeric input",
    response_model=AnomalyResponse
)
def detect_anomaly(data: InputData, request: Request):

    model = request.app.state.model_anomaly
    threshold = request.app.state.threshold_anomaly
    input_dim = request.app.state.input_dim

    # 入力次元チェック
    if len(data.values) != input_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Input dimension mismatch. Expected {input_dim}, got {len(data.values)}"
        )

    x = torch.tensor([data.values], dtype=torch.float32)

    try:
        with torch.no_grad():
            reconstructed = model(x).numpy()
    except Exception as e:
        logging.error(f"Model inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference error")

    loss = float(np.mean((np.array(data.values) - reconstructed[0]) ** 2))
    status = "abnormal" if loss > threshold else "normal"

    return {
        "anomaly_score": loss,
        "threshold": threshold,
        "status": status
    }

# ---------------------------------------------------------
# /predict
# ---------------------------------------------------------
class SensorData(BaseModel):
    temp: float = Field(..., description="温度")
    vibration: float = Field(..., description="振動")
    pressure: float = Field(..., description="圧力")

MODEL_VERSION = "v1.0.0"

@app.post(
    "/predict",
    tags=["Anomaly Detection"],
    summary="Predict anomaly from sensor data",
    response_model=PredictResponse
)
async def predict(data: SensorData, request: Request):

    model = request.app.state.model_predict
    mean = request.app.state.mean_predict
    std = request.app.state.std_predict
    threshold = request.app.state.threshold_predict

    x = np.array([[data.temp, data.vibration, data.pressure]], dtype=np.float32)
    x_norm = (x - mean) / std
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)

    try:
        with torch.no_grad():
            reconstructed = model(x_tensor).numpy()
    except Exception as e:
        logging.error(f"Predict model inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference error")

    reconstructed_denorm = reconstructed * std + mean
    loss = float(np.mean((x - reconstructed_denorm) ** 2))
    status = "normal" if loss < threshold else "abnormal"

    return {
        "anomaly_score": loss,
        "threshold": threshold,
        "status": status,
        "model_version": MODEL_VERSION,
        "timestamp": now_jst(),
        "input": data.dict()
    }

# ---------------------------------------------------------
# /health
# ---------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "timestamp": now_jst()
    }

# ---------------------------------------------------------
# /version
# ---------------------------------------------------------
API_VERSION = "1.0.0"

@app.get("/version", tags=["Info"])
async def version_info():
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "timestamp": now_jst()
    }

# ---------------------------------------------------------
# ValidationError ハンドラー
# ---------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = ".".join(str(x) for x in err["loc"] if x not in ("body",))
        errors.append({
            "field": field,
            "message": err["msg"]
        })

    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Invalid input data",
            "details": errors
        }
    )
