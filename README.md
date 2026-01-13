
# 📦 Installation

## 1. Clone
```bash
git clone https://github.com/kami263/ailab-anomaly-api
cd ailab-anomaly-api
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Run API
```bash
uvicorn main:app --reload
```

## 4. Access
- API Docs: http://localhost:8000/docs  
- Web UI: http://localhost:8000/

---

# 🔍 API Specification

## POST `/anomaly` — Detect Anomaly

### Request
```json
{
  "values": [1.2, 0.9, 1.1]
}
```

### Response
```json
{
  "score": 0.034,
  "status": "normal"
}
```

---


# 🧪 Usage Examples

## Python
```python
import requests

payload = {"values": [1.2, 0.9, 1.1]}
res = requests.post("http://localhost:8000/anomaly", json=payload)

print(res.json())
```

## cURL
```bash
curl -X POST http://localhost:8000/anomaly \
  -H "Content-Type: application/json" \
  -d '{"values":[1.2,0.9,1.1]}'
```

## JavaScript
```javascript
const res = await fetch("http://localhost:8000/anomaly", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ values: [1.2, 0.9, 1.1] })
});
console.log(await res.json());
```

---

# 🧠 Model Overview

AutoEncoder による異常検知モデル：

```
Input (x)
   │
   ▼
Encoder (Linear → ReLU)
   │
   ▼
Bottleneck (latent space)
   │
   ▼
Decoder (Linear → ReLU)
   │
   ▼
Reconstructed Output (x')
```

**Reconstruction Error = || x - x' ||**  
**Anomaly Score = Reconstruction Error**

---

# 📈 Threshold Tuning Guide

## 再構成誤差の分布を可視化
```python
import numpy as np
import matplotlib.pyplot as plt

errors = np.loadtxt("reconstruction_errors.csv")

plt.hist(errors, bins=50)
plt.show()
```

## 推奨閾値（95%）
```python
threshold = np.percentile(errors, 95)
print("Recommended threshold:", threshold)
```

---

# 🧠 Training Guide

## データ形式
```
1.2,0.9,1.1
1.0,1.1,0.95
...
```

## 学習スクリプト例
```python
import torch
from anomaly_model import AutoEncoder
import numpy as np

data = np.loadtxt("data.csv", delimiter=",")
data = torch.tensor(data, dtype=torch.float32)

model = AutoEncoder(input_dim=data.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    output = model(data)
    loss = criterion(output, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pth")
```

---

# 🧩 Customization Guide

### 入力次元を変更
```python
model = AutoEncoder(input_dim=<your_dim>)
```

### 閾値ロジックを変更
```python
status = "anomaly" if score > THRESHOLD else "normal"
```

---

# 🐳 Docker Support

## Build
```bash
docker build -t anomaly-api .
```

## Run
```bash
docker run -p 8000:8000 anomaly-api
```

---

# 📁 Project Structure

```
ailab-anomaly-api/
├── main.py
├── anomaly_model.py
├── model.pth
├── data.csv
├── templates/
├── static/
├── Dockerfile
└── requirements.txt
```

---

# 🏭 Manufacturing Use Case

```
Factory Line → Sensors → Edge Device → Anomaly Detection API
→ MES / Dashboard → Operators
```

- 振動・温度・電流・音響などのセンサー値をリアルタイム監視  
- 異常スコアが閾値を超えるとアラート発報  
- 予知保全・品質改善に活用  

---

# 📉 Performance Metrics

| 指標 | 値（例） |
|------|----------|
| 推論速度 | 1.2 ms |
| モデルサイズ | 120 KB |
| API レイテンシ | 3–5 ms |
| メモリ使用量 | ~50 MB |

---

# 🔐 Security Guide

### CORS 設定
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

# ❓ FAQ

### 入力次元が違う  
→ モデル学習時と一致しているか確認。

### model.pth が読み込めない  
→ PyTorch バージョン差異の可能性。

### 閾値はどう決める？  
→ 再構成誤差の 95〜99% を推奨。

---

# 🛠 Future Improvements

- API Key 認証  
- 閾値の自動最適化  
- マルチセンサー対応  
- オンライン学習  

---

# 📄 License



---

# 👤 Author

**kami263**  
製造業向け AI ソリューション開発

---








