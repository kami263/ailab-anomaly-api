# Anomaly Detection API

製造業向けに設計された **AutoEncoder ベースの異常検知 API** です。  
正常データのみで学習したモデルを用いて、入力データの再構成誤差から  
異常スコアを算出し、異常/正常を判定します。

FastAPI + PyTorch で構築され、REST API として簡単に統合できます。

---

## 🚀 Features

- AutoEncoder による異常検知
- 数値ベクトルを入力すると異常スコアを返す
- FastAPI による高速 API
- Docker 対応
- Web UI（templates + static）付き
- 製造業の品質管理・設備監視に最適

---

## 📦 Installation

### 1. Clone

```bash
git clone https://github.com/kami263/ailab-anomaly-api
cd ailab-anomaly-api

2. Install dependencies
pip install -r requirements.txt

3. Run API
uvicorn main:app --reload

4. Access
• 	API Docs: http://localhost:8000/docs
• 	Web UI: http://localhost:8000/

Model Overview
本 API は AutoEncoder を使用しており、
正常データのみで学習 → 再構成誤差が大きいほど異常と判定 します。
• 	入力: 数値ベクトル（例：センサー値）
• 	出力:
• 	: 再構成誤差
• 	:  or 
閾値は  内で設定されています。

 API Specification
POST  — Detect Anomaly
Request (JSON)
{
  "values": [1.2, 0.9, 1.1]
}
Response (JSON)
{
  "score": 0.034,
  "status": "normal"
}
Description
• 	入力ベクトルを AutoEncoder に通し、再構成誤差を計算
• 	閾値を超えると  を返す

🐳 Docker Support
Build
docker build -t anomaly-api .
Run






