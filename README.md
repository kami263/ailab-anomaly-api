もちろん。  
あなたが「A」と答えてくれたときにお渡しした **README.md 用の Markdown 完成版**を、もう一度まるごと出します。  
そのままコピペして README.md に貼り付ければ置き換え完了です。

---

# ✨ Anomaly Detection API  
AutoEncoder を用いた異常検知 API

このリポジトリは、AutoEncoder モデルを利用した異常検知 API を提供します。  
製造業の品質管理、設備監視、センサーデータ分析など、異常値の早期発見が必要な場面で活用できます。

---

## 🚀 特徴

- **AutoEncoder による高精度な異常検知**  
  正常データのみで学習したモデルを使用し、再構成誤差から異常スコアを算出します。
- **REST API として提供**  
  FastAPI ベースで軽量・高速。外部システムとの連携が容易。
- **Docker 対応**  
  すぐにコンテナとしてデプロイ可能。
- **シンプルな JSON インターフェース**  
  数値データを送るだけで異常スコアと判定結果を返します。

---

## 📁 プロジェクト構成

```
ailab-anomaly-api/
├── anomaly_model.py        # AutoEncoder モデルの定義と推論処理
├── model.pth               # 学習済みモデル
├── main.py                 # FastAPI アプリケーション
├── data.csv                # 学習用データ（例）
├── templates/              # Web UI 用テンプレート
├── static/                 # CSS などの静的ファイル
├── Dockerfile              # Docker イメージ構築用
├── requirements.txt        # 依存パッケージ
└── README.md               # このファイル
```

---

## 🔧 セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/kami263/ailab-anomaly-api.git
cd ailab-anomaly-api
```

### 2. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

### 3. API を起動

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

起動後、以下にアクセスできます：

- Swagger UI  
  http://localhost:8000/docs  
- ReDoc  
  http://localhost:8000/redoc

---

## 🧪 API 仕様

### POST `/anomaly` — 異常検知

#### リクエスト例

```json
{
  "values": [1.2, 0.9, 1.1, 1.0]
}
```

#### レスポンス例

```json
{
  "anomaly_score": 0.0342,
  "status": "normal"
}
```

#### パラメータ説明

| パラメータ | 型 | 説明 |
|-----------|----|------|
| values | array(float) | 数値データの配列 |

---

## 🧠 モデル概要

- **モデル:** AutoEncoder  
- **学習データ:** 正常データのみ  
- **異常判定:**  
  - 再構成誤差（MSE）を異常スコアとして使用  
  - 閾値を超えると `anomaly` と判定  

---

## 🐳 Docker での利用

### イメージをビルド

```bash
docker build -t anomaly-api .
```

### コンテナを起動

```bash
docker run -p 8000:8000 anomaly-api
```

---

## 🌐 Web UI（任意）

`/` にアクセスすると、簡易的な入力フォームが利用できます。  
ブラウザから直接異常検知を試せます。

---

## 📈 今後の改善予定

- API キー認証の追加  
- モデルの再学習 API  
- 異常スコアの可視化  
- マルチ変量データ対応  
- 時系列モデル（LSTM AutoEncoder）対応  

---

## 🤝 コントリビューション

Issue や Pull Request は歓迎します。  
改善案やバグ報告があれば気軽にどうぞ。

---

## 📄 ライセンス

MIT License

---

必要なら、  
- 英語版 README  
- 図解の追加  
- API の利用例（Python / JS）  
- バッジ（CI / Docker / License）  

なども作れます。
