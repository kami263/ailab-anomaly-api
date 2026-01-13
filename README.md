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


