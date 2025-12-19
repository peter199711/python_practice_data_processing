# 🏦 銀行客戶流失預測 (Bank Customer Churn Prediction)

使用機器學習預測銀行客戶是否會流失，並透過 SHAP 分析深入理解模型決策邏輯，最終產出可供業務單位使用的「高風險客戶名單」。

## 📋 專案概述

| 項目 | 內容 |
|------|------|
| **目標** | 預測客戶是否會流失 (Exited = 1) |
| **模型** | Random Forest Classifier |
| **資料集** | 10,000 筆銀行客戶資料 |
| **特徵數** | 19 個 (One-Hot Encoding 後) |
| **最終準確率** | 86.95% |

## 🗂️ 專案結構

```
bank_churn_prediction/
├── customer_churn_practice.ipynb  # 主要分析 Notebook
├── Customer-Churn-Records.csv     # 原始資料集
├── Bank_Churn_Prediction.db       # 輸出的 SQLite 資料庫
└── README.md                      # 專案說明文件
```

## 📊 資料集欄位說明

| 欄位名稱 | 說明 | 類型 |
|---------|------|------|
| `RowNumber` | 行號 (已移除) | int |
| `CustomerId` | 客戶 ID (已移除) | int |
| `Surname` | 姓名 (已移除) | str |
| `CreditScore` | 信用評分 | int |
| `Geography` | 國家 (France/Spain/Germany) | str → One-Hot |
| `Gender` | 性別 (Male/Female) | str → One-Hot |
| `Age` | 年齡 | int |
| `Tenure` | 往來年數 | int |
| `Balance` | 帳戶餘額 | float |
| `NumOfProducts` | 持有產品數量 | int |
| `HasCrCard` | 是否有信用卡 | int (0/1) |
| `IsActiveMember` | 是否為活躍會員 | int (0/1) |
| `EstimatedSalary` | 預估薪資 | float |
| `Exited` | **目標變數：是否流失** | int (0/1) |
| `Complain` | 是否有投訴 (洩題特徵，已移除) | int |
| `Satisfaction Score` | 滿意度評分 | int |
| `Card Type` | 卡片類型 | str → One-Hot |
| `Point Earned` | 累積點數 | int |

## 🔄 分析流程

```
1. 資料讀取與 EDA
       ↓
2. 資料清理 (移除無用欄位)
       ↓
3. 特徵工程 (One-Hot Encoding)
       ↓
4. 初版模型訓練 → 發現 Complain 是洩題特徵
       ↓
5. 移除 Complain，重新訓練
       ↓
6. 加入 class_weight='balanced' 處理不平衡資料
       ↓
7. SHAP 分析 (深度模型解釋)
       ↓
8. 調整預測門檻 (Threshold Tuning)
       ↓
9. 輸出高風險名單至 SQL 資料庫
```

## 🔍 重要發現

### 1. 洩題特徵 (Data Leakage)
初版模型準確率高達 **99.85%**，但發現 `Complain` 欄位與 `Exited` 高度相關，屬於洩題特徵。移除後準確率降至 **86.95%**，這才是模型的真實能力。

### 2. 特徵重要性排名 (SHAP 分析)

| 排名 | 特徵 | 重要性 | 解讀 |
|------|------|--------|------|
| 1 | Age | 0.22 | 年紀越大，流失風險越高 |
| 2 | NumOfProducts | 0.12 | 產品數量影響留存 |
| 3 | Balance | 0.11 | 帳戶餘額與流失相關 |
| 4 | EstimatedSalary | 0.10 | 薪資水平影響決策 |
| 5 | Point Earned | 0.10 | 累積點數反映活躍度 |

### 3. 門檻調整
將預測門檻從 50% 調整至 **30%**，以提高 Recall (召回率)：
- Recall 從 45% → **69%** (多找出更多可能流失的客戶)
- Precision 從 80% → 60% (允許部分誤報，但業務上可接受)

## 📈 模型表現

### 調整門檻後 (Threshold = 0.3)
```
              precision    recall  f1-score   support
           0       0.92      0.89      0.90      1592
           1       0.60      0.69      0.64       408

    accuracy                           0.84      2000
```

## 🛠️ 技術棧

- **Python 3.x**
- **pandas** - 資料處理
- **numpy** - 數值運算
- **scikit-learn** - 機器學習
- **matplotlib** - 資料視覺化
- **shap** - 模型可解釋性
- **sqlite3** - 資料庫輸出

## 🚀 快速開始

### 1. 安裝依賴
```bash
pip install pandas numpy scikit-learn matplotlib shap
```

### 2. 執行 Notebook
```bash
jupyter notebook customer_churn_practice.ipynb
```

### 3. 查詢高風險名單 (SQL)
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('Bank_Churn_Prediction.db')
query = """
    SELECT CustomerId, Surname, Churn_Probability
    FROM daily_churn_alert
    WHERE Churn_Probability > 0.8
    ORDER BY Churn_Probability DESC
    LIMIT 10;
"""
high_risk = pd.read_sql(query, conn)
print(high_risk)
conn.close()
```

## 📊 SHAP 視覺化說明

### Summary Plot (蜂群圖)
- **X 軸**: SHAP 值 (正值 = 推動流失，負值 = 阻止流失)
- **Y 軸**: 特徵名稱 (依重要性排序)
- **顏色**: 紅色 = 特徵值高，藍色 = 特徵值低

### Waterfall Plot (瀑布圖)
解釋單一客戶的預測結果，顯示每個特徵如何從基準值「推」或「拉」最終預測機率。

### Dependence Plot (依賴圖)
觀察單一特徵值與 SHAP 值的關係，例如：年齡越高，SHAP 值越正，表示流失風險越高。

## 💼 業務應用

此專案產出的「每日高風險客戶名單」可供：
1. **客戶關係管理 (CRM)** - 主動聯繫高風險客戶
2. **行銷活動** - 針對性優惠挽留
3. **風險監控** - 追蹤流失趨勢

## 📝 後續改進方向

- [ ] 嘗試其他模型 (XGBoost, LightGBM)
- [ ] 加入更多特徵工程
- [ ] 建立自動化排程 (每日更新名單)
- [ ] 開發 API 供業務系統串接
- [ ] A/B 測試驗證挽留策略效果

## 📄 授權

此專案僅供學習與練習使用。

---

> 📧 如有任何問題或建議，歡迎提出 Issue 或 PR！

