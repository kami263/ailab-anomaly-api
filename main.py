from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html

from app.utils.logger import setup_logging
from app.routers import anomaly, predict, system
from contextlib import asynccontextmanager
from app.services.detector import AnomalyDetector # 仮のクラス名

# 状態保持用の辞書
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時にモデルをロード
    ml_models["anomaly_detector"] = AnomalyDetector("models/model.pth")
    yield
    # 終了時のクリーンアップ（GPUメモリ解放など）
    ml_models.clear()

app = FastAPI(
    title="Anomaly Detection API",
    description="AutoEncoder を用いた異常検知 API",
    version="1.0.0",
)

# ログ設定
setup_logging()

# 静的ファイル
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    try:
        with open("app/templates/custom_swagger.html", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(html)
    except FileNotFoundError:
        return get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")


# ルーター登録
app.include_router(anomaly.router)
app.include_router(predict.router)
app.include_router(system.router)
