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

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run API
uvicorn main:app --reload

### 4. Access
• 	API Docs: http://localhost:8000/docs
• 	Web UI: http://localhost:8000/

🧠 Model Overview
本 API は AutoEncoder を使用しており、
正常データのみで学習 → 再構成誤差が大きいほど異常と判定 します。
• 	入力: 数値ベクトル（例：センサー値）
• 	出力:
• 	: 再構成誤差
• 	:  or 
閾値は  内で設定されています。

🔍  API Specification
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
docker run -p 8000:8000 anomaly-api

Project Structure
ailab-anomaly-api/
├── main.py               # FastAPI エントリポイント
├── anomaly_model.py      # AutoEncoder モデル & 推論ロジック
├── model.pth             # 学習済みモデル
├── data.csv              # サンプルデータ
├── templates/            # Web UI
├── static/               # CSS / JS
├── Dockerfile
└── requirements.txt

🛠 Future Improvements
• 	API Key 認証の追加
• 	閾値の自動最適化
• 	マルチセンサー対応
• 	モデルのオンライン学習

📄 License
MIT License（必要に応じて変更してください）

👤 Author
kami263
製造業向け AI ソリューション開発


## 📘 Architecture Diagrams

### 🧠 AutoEncoder モデル構造図
## Model Architecture (AutoEncoder)
               ┌──────────────────────────────┐
               │          Input (x)           │
               │   e.g., sensor values        │
               └──────────────┬──────────────┘
                              ▼
                     ┌────────────────┐
                     │   Encoder      │
                     │  (Linear → ReLU) 
                     └───────┬────────┘
                             ▼
                     ┌────────────────┐
                     │  Bottleneck    │
                     │ (latent space) │
                     └───────┬────────┘
                             ▼
                     ┌────────────────┐
                     │   Decoder      │
                     │ (Linear → ReLU)│
                     └───────┬────────┘
                             ▼
               ┌──────────────────────────────┐
               │     Reconstructed Output     │
               │          x' (x_hat)          │
               └──────────────────────────────┘

Reconstruction Error = || x - x' ||
Anomaly Score = Reconstruction Error

### 🔄  API フロー図（FastAPI + Model 推論
## API Flow
Client
  │
  │  POST /anomaly
  │  { "values": [...] }
  ▼
┌──────────────────────────────┐
│          FastAPI             │
│        (main.py)             │
└──────────────┬──────────────┘
               ▼
      ┌──────────────────┐
      │ anomaly_model.py │
      │  - load model    │
      │  - preprocess    │
      │  - inference     │
      │  - compute error │
      └─────────┬────────┘
                ▼
      ┌──────────────────┐
      │  Anomaly Score   │
      │  Status: normal/ │
      │          anomaly │
      └─────────┬────────┘
                ▼
Client receives JSON response

### 🧩 API 全体アーキテクチャ図（Docker も含む
## System Architecture
                 ┌──────────────────────────┐
                 │        Client App        │
                 │  (Factory system, PLC,   │
                 │   MES, dashboard, etc.)  │
                 └──────────────┬──────────┘
                                ▼
                       HTTP / JSON
                                ▼
                 ┌──────────────────────────┐
                 │        FastAPI           │
                 │        (main.py)         │
                 └──────────────┬──────────┘
                                ▼
                 ┌──────────────────────────┐
                 │   AutoEncoder Model      │
                 │   (PyTorch, model.pth)   │
                 └──────────────┬──────────┘
                                ▼
                 ┌──────────────────────────┐
                 │  Anomaly Score + Status  │
                 └──────────────────────────┘

                 (Optional)
                 ┌──────────────────────────┐
                 │        Docker            │
                 │  Containerized API       │
                 └──────────────────────────┘

## 🏭 Manufacturing Use Case

以下は、本 API が製造現場でどのように利用されるかを示すユースケース図です。
                   ┌──────────────────────────────┐
                   │        Factory Line           │
                   │  (Press, CNC, Conveyor, etc.) │
                   └───────────────┬──────────────┘
                                   ▼
                         ┌──────────────────┐
                         │   Sensors        │
                         │ (Vibration, Temp │
                         │  Current, Sound) │
                         └─────────┬────────┘
                                   ▼
                         ┌──────────────────┐
                         │  Edge Device     │
                         │ (PLC / IPC /     │
                         │  Raspberry Pi)   │
                         └─────────┬────────┘
                                   ▼
                         HTTP / JSON Request
                                   ▼
                   ┌────────────────────────────────┐
                   │     Anomaly Detection API       │
                   │        (FastAPI + AI)           │
                   │  - AutoEncoder inference        │
                   │  - Score calculation            │
                   │  - Normal / Anomaly decision    │
                   └───────────────┬────────────────┘
                                   ▼
                         JSON Response
                                   ▼
                   ┌────────────────────────────────┐
                   │     MES / Dashboard / SCADA     │
                   │  - Real-time monitoring         │
                   │  - Alerts & notifications       │
                   │  - Trend visualization          │
                   └───────────────┬────────────────┘
                                   ▼
                   ┌────────────────────────────────┐
                   │     Factory Operators           │
                   │  - Early anomaly detection      │
                   │  - Preventive maintenance       │
                   │  - Quality improvement          │
                   └────────────────────────────────┘

## ⏱️ Time-Series Flow (Sensor → API → Alert)

以下は、製造ラインでのリアルタイム異常検知の流れを示す時系列フロー図です。
Time
│
│   ① センサー計測
│   ────────────────────────────────
│      ・振動
│      ・温度
│      ・電流
│      ・音響
│
│   ② エッジデバイスでデータ収集
│   ────────────────────────────────
│      ・PLC / IPC / Raspberry Pi
│      ・サンプリング周期でデータ取得
│      ・JSON 形式に整形
│
│   ③ API に送信（HTTP POST）
│   ────────────────────────────────
│      POST /anomaly
│      { "values": [...] }
│
│   ④ AutoEncoder による推論
│   ────────────────────────────────
│      ・再構成誤差を計算
│      ・閾値と比較
│      ・異常スコア算出
│
│   ⑤ API レスポンス返却
│   ────────────────────────────────
│      {
│        "score": 0.034,
│        "status": "normal"
│      }
│
│   ⑥ ダッシュボード / MES / SCADA が受信
│   ────────────────────────────────
│      ・リアルタイム表示
│      ・トレンド更新
│
│   ⑦ アラート発報（異常時）
│   ────────────────────────────────
│      ・メール通知
│      ・ライン停止
│      ・保全担当へアラート
│
▼
End

1. 📊 サンプルコード（Python / cURL / JavaScript）
API をどう呼べばいいか、すぐ分かるようにするだけで利用者のハードルが大きく下がります。
Python Example
import requests

payload = {"values": [1.2, 0.9, 1.1]}
res = requests.post("http://localhost:8000/anomaly", json=payload)

print(res.json())


cURL Example
curl -X POST http://localhost:8000/anomaly \
  -H "Content-Type: application/json" \
  -d '{"values":[1.2,0.9,1.1]}'2. 📈 閾値のチューニング方法
製造業ユーザーは「閾値をどう決めるか」を最も気にします。
例として：
• 	再構成誤差の分布を可視化する方法
• 	95% 信頼区間を閾値にする例
• 	data.csv を使った閾値推定コード

import numpy as np

errors = np.loadtxt("reconstruction_errors.csv")
threshold = np.percentile(errors, 95)
print("Recommended threshold:", threshold)

3. 🧪 テスト方法（Unit Test / API Test）
FastAPI は pytest と相性が良いので、簡単なテスト例を載せると信頼性が上がります。

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_anomaly_api():
    res = client.post("/anomaly", json={"values": [1, 2, 3]})
    assert res.status_code == 200

    
4. 📁 モデルの再学習方法（Training Guide）
利用者が自分のデータで学習し直せるようにすると、プロジェクトの価値が一気に上がります。
例：
• 	 の追加
• 	学習データの形式
• 	学習コマンド
• 	model.pth の更新方法

python train.py --data data.csv --output model.pth

5. 🧩 API 拡張ガイド（How to Customize）
製造業の現場では「自社仕様に合わせたい」というニーズが強いので、以下のようなガイドがあると喜ばれます。
• 	入力次元を変更する方法
• 	モデル構造を変更する方法
• 	閾値ロジックを差し替える方法
• 	Web UI のカスタマイズ方法


6. 📡 本番運用ガイド（Deployment Guide）
製造業の現場では「安定稼働」が最重要。
追加すると良い内容：
• 	systemd による常駐化
• 	Nginx リバースプロキシ設定
• 	Docker Compose 例
• 	GPU 対応（任意）7. 📉 性能指標（Latency / Throughput / Model Size）
API の性能が分かると、導入判断がしやすくなります。
例：
指標      値
推論速度  1.2 ms / request
モデルサイズ  120 KB
API レイテンシ 3–5 ms
メモリ使用量  ~50 MB

8. 🔐 セキュリティガイド
製造業ではセキュリティ要件が厳しいため、以下を追加すると信頼性が高まります。
• 	API Key 認証の例
• 	HTTPS 化の方法
• 	CORS 設定
• 	ログ管理（PII を含まない）


9. 📚 FAQ（よくある質問）
ユーザーがつまずきやすいポイントをまとめると親切です。
例：
• 	入力次元が違うと言われる
• 	model.pth が読み込めない
• 	Docker で GPU を使いたい
• 	閾値をどう決める？


10. 🧵 実運用シナリオ（Case Studies）
製造業向けなので、具体例があると説得力が増します。
例：
• 	CNC 加工機の振動異常検知
• 	モーター電流の異常検知
• 	温度センサーのドリフト検知
• 	ベアリング故障の早期兆候検知




  


