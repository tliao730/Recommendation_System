# DSCI 553 Competition — Yelp Rating Prediction

## Environment

- **Local**: `/opt/anaconda3/bin/python3` (Python 3.13, xgboost 3.1.1)
- **Vocareum**: Python 3.6, xgboost 0.x (old API — use `objective="reg:linear"`, NOT `reg:squarederror`)
- **Data dir**: `data/`
- **Submission file**: `src/competition.py`
- **Analysis script**: `src/feature_analysis.py` — run with `--no-wandb` flag

## Current Approach (competition.py)

### Features (131 total)

| Block | Source | Count |
|---|---|---|
| Business features | business.json + checkin.json + tip.json | 104 |
| User features | user.json + tip.json | 24 |
| u_avg_train (OOF) | yelp_train.csv, 5-fold out-of-fold | 1 |
| i_avg_train (OOF) | yelp_train.csv, 5-fold out-of-fold | 1 |
| diff = u_avg - i_avg | derived | 1 |

**OOF 的原因**：直接用全訓練集計算 u_avg/i_avg 會把當前 row 的 label 包進去（data leakage），導致 val RMSE 爆掉（1.00+）但 test 正常。OOF 版本消除 leakage，val RMSE 改善。

### XGBoost 參數

```python
max_depth=8, learning_rate=0.05, n_estimators=474,
subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
gamma=0.1, reg_alpha=0.1, reg_lambda=2.0,
objective="reg:linear",  # Vocareum 舊版不支援 reg:squarederror
```

`n_estimators=474`：用 131-feat OOF 版本 + early stopping (patience=50) 在 yelp_val.csv 上找到。

### CF（Item-based Collaborative Filtering）

- Pearson similarity + baseline adjustment
- 鄰居數 ≥ 15：CF 權重 0.25
- 鄰居數 ≥ 5：CF 權重 0.15
- 鄰居數 < 5：純 XGBoost

## Vocareum 實驗紀錄

| 版本 | 說明 | Vocareum RMSE |
|---|---|---|
| competition_unofficial_script.py | 131 feat (leaky) + CF 0.35/0.20/0.05 | **0.9751** |
| competition.py + MF blend | 128 feat + Matrix Factorization (未收斂) | 0.9791 |
| competition.py (OOF 131 feat) | 目前版本，待測 | 0.9747 |

## 已嘗試但無效的方法

| 方法 | 結果 | 原因 |
|---|---|---|
| Matrix Factorization (SGD biased SVD) | val RMSE 1.00+，Vocareum 0.9791（更差）| MF 在稀疏 Yelp 資料上嚴重過擬合 |
| Leaky u_avg/i_avg（直接加入 X_train）| val RMSE 1.01+（爆掉）| data leakage，model 記住 label |
| LOO averages（leave-one-out）| val RMSE 0.98046（比 128 feat 差）| 低頻 user/business 的 LOO 值太雜訊 |
| interaction features（b_stars × u_avg_stars 等）| val RMSE 0.97786（比 128 feat 差）| XGBoost 深度節點已隱式捕捉交互 |
| StandardScaler on top-10 features | 無效 | Tree-based 模型對 monotonic 縮放不敏感 |

## 最重要的特徵（128 feat 版本）

1. `b_stars`（business 整體平均）— 最重要，重要度 0.28
2. `u_avg_stars`（user 整體平均）— 第二，重要度 0.096
3. `b_restaurantsgoodforgroups`, `b_cat_restaurants`, `b_hastv` 等業務屬性

## 目前最佳 val RMSE

- 128 feat + 最佳 CF 權重：**0.97496**
- 131 feat OOF + early stopping n=474 + 最佳 CF：**0.97465**

## 同學成績參考

- 同學最佳：RMSE = 0.9333（推測使用優化的協同過濾或 neural CF）
- 我們目前最佳 Vocareum：0.9747
