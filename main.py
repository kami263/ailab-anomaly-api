import numpy as np
import pandas as pd
import torch
import os
import logging
import json
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from pydantic import BaseModel, Field
from typing import Dict

from anomaly_model import (
    AutoEncoder,
    train_autoencoder_from_csv,
    load_model_and_threshold
)

# ---------------------------------------------------------
# JSON ログフォーマッタ（安全版）
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

# uvicorn ロガーに安全に適用
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        logger.handlers[0].setFormatter(JsonFormatter())

# ---------------------------------------------------------
# FastAPI アプリ
# ---------------------------------------------------------
app = FastAPI(
    title="Anomaly Detection API",
    description = """
本 API は、AutoEncoder を用いた異常検知モデルにより、
設備データ・センサーデータから異常スコアを算出し、
正常／異常をリアルタイムに判定するためのエンドポイントを提供します。

## 主な機能
- 数値配列からの異常スコア算出（/anomaly）
- センサーデータ（温度・振動・圧力）からの異常判定（/predict）
- API 稼働状況の確認（/health）
- API / モデルバージョンの取得（/version）

## 特徴
- AutoEncoder による再構成誤差を用いた高精度な異常検知
- 入力値の正規化・復元処理を含む一貫した推論パイプライン
- 統一されたレスポンスモデル（ResponseModel）
- 統一された ValidationError（422）レスポンス
- Docker コンテナとして即時デプロイ可能

## 想定ユースケース
- 製造業の設備監視
- IoT センサーデータの異常検知
- 予兆保全（Predictive Maintenance）
- 異常スコアの可視化・ダッシュボード連携

""",
    version="1.0.0"
)
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    with open("templates/custom_swagger.html") as f:
        html = f.read()
    return HTMLResponse(html)
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
# /anomaly 用モデル
# ---------------------------------------------------------
CSV_PATH = "/app/data.csv"
MODEL_PATH = "/app/model.pth"

df = pd.read_csv(CSV_PATH)
input_dim = df.select_dtypes(include=["number"]).shape[1]

model_anomaly = AutoEncoder()

if os.path.exists(MODEL_PATH):
    print("Loading saved model for /anomaly ...")
    model_anomaly.load_state_dict(torch.load(MODEL_PATH))
    model_anomaly.eval()
else:
    print("Training model for /anomaly ...")
    model_anomaly, mean, std = train_autoencoder_from_csv(CSV_PATH)

class InputData(BaseModel):
    values: list[float]

@app.post(
    "/anomaly",
    tags=["Anomaly Detection"],
    summary="Detect anomaly from numeric input",
    operation_id="detect_anomaly",
    response_model=AnomalyResponse
)
def detect_anomaly(data: InputData):
    x = torch.tensor([data.values], dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model_anomaly(x).numpy()

    loss = float(np.mean((np.array(data.values) - reconstructed[0]) ** 2))

    THRESHOLD = 1.0
    status = "abnormal" if loss > THRESHOLD else "normal"

    return {
        "anomaly_score": loss,
        "threshold": THRESHOLD,
        "status": status
    }

# ---------------------------------------------------------
# /predict 用モデル
# ---------------------------------------------------------
class SensorData(BaseModel):
    temp: float = Field(..., description="温度")
    vibration: float = Field(..., description="振動")
    pressure: float = Field(..., description="圧力")

model, mean, std, threshold = load_model_and_threshold()

MODEL_VERSION = "v1.0.0"

@app.post(
    "/predict",
    tags=["Anomaly Detection"],
    summary="Predict anomaly from sensor data",
    operation_id="predict_sensor",
    response_model=PredictResponse
)
async def predict(data: SensorData):
    x = np.array([[data.temp, data.vibration, data.pressure]], dtype=np.float32)

    x_norm = (x - mean) / std
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(x_tensor).numpy()

    reconstructed_denorm = reconstructed * std + mean
    loss = float(np.mean((x - reconstructed_denorm) ** 2))

    status = "normal" if loss < threshold else "abnormal"

    return {
        "anomaly_score": loss,
        "threshold": threshold,
        "status": status,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "input": data.dict()
    }

# ---------------------------------------------------------
# /health
# ---------------------------------------------------------
@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    operation_id="health_check"
)
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat()
    }

# ---------------------------------------------------------
# /version
# ---------------------------------------------------------
API_VERSION = "1.0.0"

@app.get(
    "/version",
    tags=["Info"],
    summary="Version Info",
    operation_id="version_info"
)
async def version_info():
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat()
    }

# ---------------------------------------------------------
# STEP4: ValidationError の統一ハンドラー
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
