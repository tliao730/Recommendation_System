# DSCI 553 Competition — Yelp Rating Prediction

## Environment

- **Local**: `/opt/anaconda3/bin/python3` (Python 3.13, xgboost 3.1.1)
- **Vocareum**: Python 3.6, xgboost 0.x (old API)
- **Data dir**: `data/`
- **Submission file**: `src/competition.py`
- **Tuning script**: `src/tune_xgb.py` — offline hyperparameter search, no Spark needed

## Current Approach (competition.py)

### Features (144 total)

| Block | Source | Count |
|---|---|---|
| Business features | business.json + checkin.json + tip.json | 104 |
| Business photo features | photo.json (log1p count + 5 label ratios) | 6 |
| Business tip content | tip.json (avg word count, avg exclamation) | 2 |
| User features | user.json + tip.json | 24 |
| User tip content | tip.json (avg word count) | 1 |
| u_avg_train (OOF) | yelp_train.csv, 5-fold out-of-fold | 1 |
| i_avg_train (OOF) | yelp_train.csv, 5-fold out-of-fold | 1 |
| diff = u_avg - i_avg | derived | 1 |
| ucat_avg (OOF) | user's avg rating per matching category, OOF | 1 |
| jaccard | user category set ∩ business category set | 1 |
| ub_tip_flag | implicit feedback: user tipped this business | 1 |
| svd_score (OOF) | 5-fold biased SVD via SGD | 1 |

**OOF 的原因**：直接用全訓練集計算 u_avg/i_avg 會把當前 row 的 label 包進去（data leakage），導致 val RMSE 爆掉（1.00+）。OOF 版本用其他 4 fold 計算，消除 leakage。SVD 特徵同樣做 OOF 處理。

### XGBoost 參數

```python
max_depth=8, learning_rate=0.05, n_estimators=372,
subsample=0.85, colsample_bytree=0.6, min_child_weight=10,
gamma=0.0, reg_alpha=0.5, reg_lambda=2.0,
objective="reg:linear",  # Vocareum 舊版不支援 reg:squarederror
nthread=4, seed=42
```

- `n_estimators=372`：用 132-feat OOF SVD 版本 + early stopping (patience=50) 在 yelp_val.csv 上找到；144-feat 版本沿用，尚未重新 tune。
- 加入 SVD 特徵後最佳 n 從 474 降到 372（SVD 提供額外信號，收斂更快）。
- `colsample_bytree=0.6`, `min_child_weight=10`, `reg_alpha=0.5`：tune_xgb.py 30-trial random search 找到的更強正規化組合。

### OOF SVD（biased SVD via mini-batch SGD）

```python
_MF_FACTORS = 10   # latent factor 數
_MF_EPOCHS  = 8    # per OOF fold；final model 用 12 epochs
```

5-fold OOF：每 fold 用 80% 訓練資料訓練 SVD，預測 20% hold-out，避免 leakage。
Test 時用全量訓練資料訓練 final SVD。

### CF（Item-based Collaborative Filtering）

- Pearson similarity + baseline adjustment（baseline = 0.15×u_avg + 0.85×i_avg）
- Shrinkage 正規化：sim *= |共同評分數| / (|共同評分數| + 2)
- 鄰居數 ≥ 15：CF 權重 **0.15**
- 鄰居數 ≥ 5：CF 權重 **0.05**
- 鄰居數 < 5：純 XGBoost

### Tuning Flags（competition.py 頂部）

```python
_USE_OOF_CF = False  # OOF CF 作為特徵（leakage 問題，設 False）
_USE_SVD    = True   # OOF SVD 作為特徵
```

## Vocareum 實驗紀錄

| 版本 | 說明 | Vocareum RMSE |
|---|---|---|
| competition_unofficial_script.py | 131 feat (leaky) + CF 0.35/0.20/0.05 | 0.9751 |
| competition.py + MF blend | 128 feat + Matrix Factorization (未收斂) | 0.9791 |
| competition.py (OOF 131 feat) | OOF u_avg/i_avg, n=474, lr=0.05 | 0.9747 |
| competition.py (132 feat) | 132 feat + OOF SVD, n=372, lr=0.05 + CF blend | 0.9742 |
| competition.py (164 feat，失敗) | +P_u/Q_i 原始向量（空間不兼容）→ timeout | 2.94（miss:125789） |
| **competition.py（目前最佳）** | 144 feat + photo/tip/ucat/jaccard/ub_tip, n=372 | **0.9726** |

## 已嘗試但無效的方法

| 方法 | 結果 | 原因 |
|---|---|---|
| Matrix Factorization (SGD biased SVD) 獨立模型 | val RMSE 1.00+，Vocareum 0.9791 | MF 在稀疏 Yelp 資料上嚴重過擬合 |
| Leaky u_avg/i_avg（直接加入 X_train） | val RMSE 1.01+（爆掉） | data leakage，XGBoost 記住 label |
| LOO averages（leave-one-out） | val RMSE 0.98046 | 低頻 user/business 的 LOO 值太雜訊 |
| interaction features（b_stars × u_avg_stars 等） | val RMSE 0.97786 | XGBoost 深度節點已隱式捕捉交互 |
| StandardScaler on top-10 features | 無效 | Tree-based 模型對 monotonic 縮放不敏感 |
| OOF CF 作為特徵（approximate，full i2u） | val RMSE 1.097 | Pearson sim 仍用 full i2u，leakage 未完全消除 |
| n=1359, lr=0.02（更多樹、低學習率） | local 0.9733，Vocareum 崩潰 | 樹太多，超過 Vocareum 記憶體/時間限制 |

## 最重要的特徵

1. `b_stars`（business 整體平均）— 最重要，重要度 0.28
2. `u_avg_stars`（user 整體平均）— 第二，重要度 0.096
3. `b_restaurantsgoodforgroups`, `b_cat_restaurants`, `b_hastv` 等業務屬性
4. `svd_score`（OOF SVD 預測值）— 新增

## val RMSE 演進

| 版本 | local val RMSE | Vocareum RMSE |
|---|---|---|
| 128 feat + CF 0.25/0.15 | 0.97496 | 0.9751 |
| 131 feat OOF + n=474 + CF 0.25/0.15 | 0.97465 | 0.9747 |
| 132 feat OOF SVD + n=372 + CF 0.15/0.05 | 0.9739 | 0.9742 |
| **144 feat + photo/tip/ucat/jaccard/ub_tip** | **0.9726** | **0.9726** |

## 下一步可能的改善方向

| 方法 | 預估改善 | 複雜度 |
|---|---|---|
| re-tune n_estimators（針對 144-feat）| 可能 −0.001～−0.002 | 極低（跑 tune_xgb.py） |
| ALS Matrix Factorization（取代 SGD MF） | −0.003～−0.008 | 中 |
| 社交特徵（朋友對同一 business 的評分） | −0.002～−0.004 | 中 |

## 環境注意事項

- **Vocareum** 使用 `objective="reg:linear"`（舊版不支援 `reg:squarederror`）
- **Local** 測試時設 `JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.17/libexec/openjdk.jdk/Contents/Home`
- **tune_xgb.py** 在 local 用 `objective="reg:squarederror"`，competition.py 用 `"reg:linear"`
- n=1359 在 Vocareum 會崩潰（記憶體/時間超限），安全上限約 n=500～600

## 同學成績參考

- 同學最佳：RMSE = 0.9333（推測使用 LightGCN / Neural CF / ALS+implicit）
- 我們目前最佳 Vocareum：**0.9726**
