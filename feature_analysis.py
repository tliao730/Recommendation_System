#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature engineering analysis for DSCI 553 Competition.
Trains on yelp_train.csv, validates on yelp_val.csv.
Logs per-round train/val RMSE curves + feature importances to Weights & Biases.

Usage:
    /opt/anaconda3/bin/python3 src/feature_analysis.py data/ \
        [--project yelp-competition] [--run baseline-vs-full]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error)
from sklearn.preprocessing import StandardScaler

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

_USE_WANDB = False   # set at runtime via --no-wandb

# ── Constants (mirrors competition.py) ─────────────────────────────────────────

CURRENT_YEAR = 2026

_TOP_STATES = ["AZ", "NV", "ON", "NC", "OH", "PA", "QC", "AB", "WI", "IL"]

_TOP_CATS = [
    "Restaurants", "Shopping", "Food", "Beauty & Spas", "Home Services",
    "Health & Medical", "Local Services", "Automotive", "Nightlife", "Bars",
    "Event Planning & Services", "Active Life", "Fashion", "Coffee & Tea",
    "Sandwiches", "Hair Salons", "Fast Food", "American (Traditional)",
    "Pizza", "Home & Garden", "Hotels & Travel", "Arts & Entertainment",
    "Burgers", "Mexican", "Chinese", "Italian", "Japanese", "Sushi Bars",
    "Breakfast & Brunch", "Grocery",
]

_BOOL_ATTRS = [
    "BikeParking", "BusinessAcceptsCreditCards", "Caters", "CoatCheck",
    "DogsAllowed", "DriveThru", "GoodForDancing", "GoodForKids", "HappyHour",
    "HasTV", "Open24Hours", "OutdoorSeating", "RestaurantsDelivery",
    "RestaurantsGoodForGroups", "RestaurantsReservations", "RestaurantsTableService",
    "RestaurantsTakeOut", "WheelchairAccessible", "BYOB", "ByAppointmentOnly",
    "Corkage", "AcceptsInsurance",
]

_NOISE   = {"quiet": 0.0, "average": 1.0, "loud": 2.0, "very_loud": 3.0}
_ATTIRE  = {"casual": 0.0, "dressy": 1.0, "formal": 2.0}
_AMBIENCE = ["romantic", "intimate", "classy", "hipster", "touristy",
             "trendy", "upscale", "casual"]
_PARKING  = ["garage", "street", "validated", "lot", "valet"]
_MEAL     = ["dessert", "latenight", "lunch", "dinner", "breakfast", "brunch"]
_DAYS     = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_COMPS    = [
    "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute",
    "compliment_list", "compliment_note", "compliment_plain", "compliment_cool",
    "compliment_funny", "compliment_writer", "compliment_photos",
]

# ── Feature name list (for wandb importance chart) ─────────────────────────────

def _build_feat_names():
    biz = (
        ["b_stars", "b_review_count", "b_is_open", "b_latitude", "b_longitude"]
        + [f"b_{a.lower()}" for a in _BOOL_ATTRS]
        + ["b_price_range", "b_alc_none", "b_alc_beer_wine", "b_alc_full_bar",
           "b_wifi_no", "b_wifi_free", "b_wifi_paid", "b_noise", "b_attire"]
        + [f"b_amb_{k}" for k in _AMBIENCE]
        + [f"b_park_{k}" for k in _PARKING]
        + [f"b_meal_{k}" for k in _MEAL]
        + [f"b_hrs_{d[:3].lower()}" for d in _DAYS]
        + [f"b_state_{s}" for s in _TOP_STATES]
        + [f"b_cat_{c.replace(' ','_').replace('&','and').replace('(','').replace(')','')[:14].lower()}"
           for c in _TOP_CATS]
        + ["b_checkin_cnt", "b_tip_cnt"]
    )
    usr = (
        ["u_review_count", "u_avg_stars", "u_fans",
         "u_useful", "u_funny", "u_cool",
         "u_tenure", "u_friend_log", "u_is_elite", "u_elite_yrs"]
        + [f"u_{k[len('compliment_'):]}" for k in _COMPS]
        + ["u_total_comps", "u_engagement", "u_tip_cnt"]
    )
    return biz + usr   # 104 + 24 = 128

FEAT_NAMES = _build_feat_names()

# ── Continuous feature indices (for StandardScaler, within the 128-dim vector) ──
# Business layout:
#   [0]     stars           [1]  review_count   [2]  is_open (binary)
#   [3]     latitude        [4]  longitude
#   [5..26] 22 bool attrs (binary 0/1/-1)
#   [27]    price_range (ordinal)
#   [28..30] alcohol one-hot  [31..33] wifi one-hot  (binary)
#   [34]    noise (ordinal)   [35]  attire (ordinal)
#   [36..43] ambience  [44..48] parking  [49..54] meal  (binary -1/0/1)
#   [55..61] 7 hours (continuous, -1 if closed)
#   [62..71] 10 state one-hot   [72..101] 30 cat one-hot  (binary)
#   [102]   checkin_cnt         [103]     tip_biz_cnt
# User layout (offset 104):
#   [104]   review_count  [105]  avg_stars   [106]  fans
#   [107]   useful        [108]  funny        [109]  cool
#   [110]   tenure        [111]  friend_log   [112]  is_elite (binary)
#   [113]   elite_yrs     [114..124] 11 compliments
#   [125]   total_comps   [126]  engagement   [127]  tip_usr_cnt
CONTINUOUS_IDX = (
    [0, 1, 3, 4, 27, 34, 35]            # stars, rev_cnt, lat, lon, price, noise, attire
    + list(range(55, 62))               # hours Mon-Sun
    + [102, 103]                        # checkin, tip_biz
    + list(range(104, 112))             # u: rev_cnt, avg_stars, fans, useful, funny, cool, tenure, friend_log
    + [113]                             # u: elite_yrs
    + list(range(114, 128))             # u: 11 comps + total_comps + engagement + tip_usr
)

# ── Interaction feature names (appended after the 128 base features) ───────────
INTERACT_NAMES = ["x_stars_x_uavg", "x_u_bias", "x_b_bias", "x_bias_cross"]


def add_interactions(X, g_avg):
    """
    Append 4 interaction columns to X (shape N x 128+) in-place-free.
      b_stars    = X[:, 0]      (business avg rating)
      u_avg      = X[:, 105]    (user avg rating)
      g_avg      = global train mean rating
    Appended cols:
      b_stars * u_avg            -- joint quality signal
      u_avg - g_avg              -- how generous/harsh this user rates (user bias)
      b_stars - g_avg            -- how good/bad this business is vs average (biz bias)
      (b_stars - g_avg) * (u_avg - g_avg)  -- bias interaction
    """
    b_stars  = X[:, 0:1]
    u_avg    = X[:, 105:106]
    b_bias   = b_stars - g_avg
    u_bias   = u_avg   - g_avg
    return np.hstack([X,
                      b_stars * u_avg,   # cross
                      u_bias,
                      b_bias,
                      b_bias * u_bias])  # N x 132


# ── Feature extraction (pure Python, no Spark needed for local analysis) ────────

def _hours(h_dict, day):
    if not h_dict or day not in h_dict:
        return -1.0
    try:
        op, cl = h_dict[day].split("-")
        oh, om = map(int, op.split(":"))
        ch, cm = map(int, cl.split(":"))
        d = (ch + cm / 60.0) - (oh + om / 60.0)
        return float(d + 24 if d < 0 else d)
    except Exception:
        return -1.0


def _da(s, k):
    if f"'{k}': True" in s:
        return 1.0
    if f"'{k}': False" in s:
        return 0.0
    return -1.0


def _biz_feat(d):
    a = d.get("attributes") or {}
    f = [
        float(d.get("stars", 3.5) or 3.5),
        float(d.get("review_count", 0) or 0),
        float(d.get("is_open", 1) or 1),
        float(d.get("latitude",  0.0) or 0.0),
        float(d.get("longitude", 0.0) or 0.0),
    ]
    for b in _BOOL_ATTRS:
        v = a.get(b)
        f.append(1.0 if v == "True" else (0.0 if v == "False" else -1.0))
    pr = a.get("RestaurantsPriceRange2")
    f.append(float(pr) if pr in ("1", "2", "3", "4") else -1.0)
    alc  = a.get("Alcohol", "")
    wifi = a.get("WiFi", "")
    f += [1.0 if alc == "none" else 0.0,
          1.0 if alc == "beer_and_wine" else 0.0,
          1.0 if alc == "full_bar" else 0.0,
          1.0 if wifi == "no" else 0.0,
          1.0 if wifi == "free" else 0.0,
          1.0 if wifi == "paid" else 0.0]
    f.append(_NOISE.get(str(a.get("NoiseLevel", "")), -1.0))
    f.append(_ATTIRE.get(str(a.get("RestaurantsAttire", "")), -1.0))
    for keys, field in [(_AMBIENCE, "Ambience"), (_PARKING, "BusinessParking"), (_MEAL, "GoodForMeal")]:
        fs = str(a.get(field, ""))
        for k in keys:
            f.append(_da(fs, k))
    h = d.get("hours")
    for day in _DAYS:
        f.append(_hours(h, day))
    state = d.get("state", "")
    for s in _TOP_STATES:
        f.append(1.0 if state == s else 0.0)
    cats = d.get("categories") or ""
    for c in _TOP_CATS:
        f.append(1.0 if c in cats else 0.0)
    return f   # 102 floats


def _usr_feat(d):
    since = d.get("yelping_since", "")
    try:
        tenure = float(CURRENT_YEAR - int(since[:4]))
    except Exception:
        tenure = 10.0
    fr = d.get("friends", "")
    fc = math.log1p(len(fr.split(",")) if (fr and fr != "None") else 0)
    el    = d.get("elite", "")
    is_el = 1.0 if (el and el != "None") else 0.0
    el_yrs = float(len(el.split(","))) if (el and el != "None") else 0.0
    us = float(d.get("useful", 0) or 0)
    fn = float(d.get("funny",  0) or 0)
    cl = float(d.get("cool",   0) or 0)
    comps = [float(d.get(k, 0) or 0) for k in _COMPS]
    return [
        float(d.get("review_count",   0)   or 0),
        float(d.get("average_stars", 3.5)  or 3.5),
        float(d.get("fans",           0)   or 0),
        us, fn, cl, tenure, fc, is_el, el_yrs,
    ] + comps + [sum(comps), us + fn + cl]   # 23 floats


# ── Column-wise mean default (exclude -1 sentinel) ─────────────────────────────

def _col_defaults(rows, n):
    s = [0.0] * n
    c = [0]   * n
    for row in rows:
        for i, v in enumerate(row):
            if v != -1.0:
                s[i] += v
                c[i] += 1
    return [s[i] / c[i] if c[i] > 0 else 0.0 for i in range(n)]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all(folder):
    print("[load] business.json ...", end=" ", flush=True)
    t = time.time()
    biz_raw = {}
    with open(os.path.join(folder, "business.json")) as f:
        for line in f:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            biz_raw[bid] = _biz_feat(d)
    print(f"{len(biz_raw):,} rows  ({time.time()-t:.1f}s)")

    print("[load] user.json ...", end=" ", flush=True)
    t = time.time()
    usr_raw = {}
    with open(os.path.join(folder, "user.json")) as f:
        for line in f:
            d   = json.loads(line)
            uid = d.get("user_id", "")
            usr_raw[uid] = _usr_feat(d)
    print(f"{len(usr_raw):,} rows  ({time.time()-t:.1f}s)")

    print("[load] checkin.json ...", end=" ", flush=True)
    t = time.time()
    ck_map = {}
    with open(os.path.join(folder, "checkin.json")) as f:
        for line in f:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            td  = d.get("time", {})
            ck_map[bid] = float(sum(td.values())) if isinstance(td, dict) else 0.0
    print(f"{len(ck_map):,} rows  ({time.time()-t:.1f}s)")

    print("[load] tip.json ...", end=" ", flush=True)
    t = time.time()
    tb_map, tu_map = {}, {}
    with open(os.path.join(folder, "tip.json")) as f:
        for line in f:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            uid = d.get("user_id", "")
            tb_map[bid] = tb_map.get(bid, 0.0) + 1.0
            tu_map[uid] = tu_map.get(uid, 0.0) + 1.0
    print(f"b:{len(tb_map):,}  u:{len(tu_map):,}  ({time.time()-t:.1f}s)")

    biz_map = {bid: f + [ck_map.get(bid, 0.0), tb_map.get(bid, 0.0)]
               for bid, f in biz_raw.items()}
    usr_map = {uid: f + [tu_map.get(uid, 0.0)]
               for uid, f in usr_raw.items()}

    biz_vals = list(biz_map.values())
    usr_vals = list(usr_map.values())
    b_default = _col_defaults(biz_vals, len(biz_vals[0]) if biz_vals else 104)
    u_default = _col_defaults(usr_vals, len(usr_vals[0]) if usr_vals else 24)

    return biz_map, usr_map, b_default, u_default


def load_csv(path, has_label=True):
    rows = []
    with open(path) as f:
        header = next(f)
        for line in f:
            parts = line.strip().split(",")
            if has_label:
                rows.append((parts[0], parts[1], float(parts[2])))
            else:
                rows.append((parts[0], parts[1]))
    return rows


def build_xy(rows, biz_map, usr_map, b_default, u_default, has_label=True):
    X, y = [], []
    for row in rows:
        uid, bid = row[0], row[1]
        feat = list(biz_map.get(bid, b_default)) + list(usr_map.get(uid, u_default))
        X.append(feat)
        if has_label:
            y.append(row[2])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) if has_label else None


# ── Baseline feature builder (13 features, matches task2_2.py) ─────────────────

def build_xy_baseline(rows, biz_map, usr_map, has_label=True):
    """
    6 business: stars, review_count, price, is_open, good_for_kids, delivery
    7 user:     review_count, useful, fans, avg_stars, elite_count, account_age, compliments
    """
    X, y = [], []
    for row in rows:
        uid, bid = row[0], row[1]
        bd = biz_map.get(bid)
        if bd is not None:
            b_feats = [bd[0], bd[1], bd[28], bd[2],   # stars, rev_cnt, price_range, is_open
                       bd[_BOOL_ATTRS.index("GoodForKids") + 5],
                       bd[_BOOL_ATTRS.index("RestaurantsDelivery") + 5]]
        else:
            b_feats = [3.5, 50.0, 2.0, 1.0, 0.0, 0.0]

        ud = usr_map.get(uid)
        if ud is not None:
            # indices: 0=rev_cnt, 3=useful, 2=fans, 1=avg_stars, 9=elite_yrs, 6=tenure, 21=total_comps
            u_feats = [ud[0], ud[3], ud[2], ud[1], ud[9], ud[6], ud[21]]
        else:
            u_feats = [20.0, 5.0, 1.0, 3.7, 0.0, 8.0, 10.0]

        X.append(b_feats + u_feats)
        if has_label:
            y.append(row[2])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) if has_label else None


# ── Metrics ────────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, prefix=""):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    y_pr = np.clip(np.round(y_pred).astype(int), 1, 5)
    y_tr = np.clip(np.round(y_true).astype(int), 1, 5)
    acc  = accuracy_score(y_tr, y_pr)
    f1   = f1_score(y_tr, y_pr, average="weighted", labels=[1, 2, 3, 4, 5], zero_division=0)
    return {f"{prefix}rmse": round(rmse, 5),
            f"{prefix}mae":  round(mae,  5),
            f"{prefix}acc":  round(acc,  5),
            f"{prefix}f1":   round(f1,   5)}


# ── WandbXGBCallback: log per-round metrics ────────────────────────────────────

class WandbXGBCallback(xgb.callback.TrainingCallback):
    """Log per-round XGBoost metrics to wandb. Keys: validation_0=train, validation_1=val."""
    def after_iteration(self, model, epoch, evals_log):
        if not _USE_WANDB:
            return False
        # sklearn eval_set auto-names: validation_0 -> train, validation_1 -> val
        rename = {"validation_0": "train", "validation_1": "val"}
        log = {}
        for dataset, metric_dict in evals_log.items():
            label = rename.get(dataset, dataset)
            for metric, values in metric_dict.items():
                log[f"{label}_{metric}"] = values[-1]
        wandb.log(log, step=epoch)
        return False


# ── Train one XGBoost experiment ───────────────────────────────────────────────

XGB_BASE_PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective="reg:squarederror",
    n_jobs=4,
    verbosity=0,
    seed=42,
)


def run_experiment(name, X_tr, y_tr, X_val, y_val, feat_names=None, extra_params=None):
    params = dict(XGB_BASE_PARAMS)
    if extra_params:
        params.update(extra_params)

    # XGBoost 3.x: callbacks go to the constructor, not fit()
    # eval_set format: [(X, y), ...] — sklearn API, no named tuples
    reg = xgb.XGBRegressor(**params, callbacks=[WandbXGBCallback()])
    reg.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
    )

    tr_metrics  = metrics(y_tr,  reg.predict(X_tr),  prefix="train_")
    val_metrics = metrics(y_val, reg.predict(X_val), prefix="val_")
    all_m = {**tr_metrics, **val_metrics}
    if _USE_WANDB:
        wandb.log(all_m)

    print(f"\n  [{name}]")
    for k, v in all_m.items():
        print(f"    {k:25s}: {v}")

    if feat_names is not None and hasattr(reg, "feature_importances_"):
        fi = reg.feature_importances_
        top_k = 30
        top_idx = np.argsort(fi)[::-1][:top_k]

        if _USE_WANDB:
            table = wandb.Table(
                columns=["feature", "importance"],
                data=[[feat_names[i], float(fi[i])] for i in top_idx]
            )
            wandb.log({
                f"{name}_feature_importance": wandb.plot.bar(
                    table, "feature", "importance",
                    title=f"{name} - Top {top_k} Feature Importances"
                )
            })

        print(f"\n  Top-10 features:")
        for i in top_idx[:10]:
            print(f"    {feat_names[i]:35s}: {fi[i]:.4f}")

    return reg, all_m


# ── Ablation: drop low-importance features ─────────────────────────────────────

def ablation_threshold(reg_full, X_tr, y_tr, X_val, y_val, feat_names, threshold=0.001):
    fi = reg_full.feature_importances_
    keep = np.where(fi >= threshold)[0]
    drop = np.where(fi < threshold)[0]
    print(f"\n  [Ablation] threshold={threshold}: keep {len(keep)} / drop {len(drop)} features")
    if len(drop) == 0:
        print("  No features dropped.")
        return

    X_tr_r  = X_tr[:, keep]
    X_val_r = X_val[:, keep]
    kept_names = [feat_names[i] for i in keep]

    _, abl_m = run_experiment(
        f"ablation_thr{threshold}",
        X_tr_r, y_tr, X_val_r, y_val,
        feat_names=kept_names,
    )

    if _USE_WANDB:
        wandb.log({"ablation_features_kept": len(keep),
                   "ablation_val_rmse": abl_m["val_rmse"]})


# ── Hyperparameter tuning ──────────────────────────────────────────────────────

def param_sweep(X_tr, y_tr, X_vl, y_vl, base_params):
    """
    Phase 1: grid search over max_depth x learning_rate (fixed n_estimators).
    Phase 2: best combo + early stopping to find optimal n_estimators.
    Returns best_params dict.
    """
    depths = [4, 5, 6, 7, 8]
    lrs    = [0.03, 0.05, 0.10]

    print(f"\n  Grid: max_depth={depths} x learning_rate={lrs}")
    print(f"  {'max_depth':>10} {'lr':>6} {'train_rmse':>12} {'val_rmse':>12} {'n_est':>6}")
    print("  " + "-" * 50)

    results = []
    for depth in depths:
        for lr in lrs:
            p = dict(base_params, max_depth=depth, learning_rate=lr, n_estimators=400)
            reg = xgb.XGBRegressor(**p)
            reg.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            tr_rmse  = math.sqrt(mean_squared_error(y_tr, reg.predict(X_tr)))
            val_rmse = math.sqrt(mean_squared_error(y_vl, reg.predict(X_vl)))
            results.append((val_rmse, depth, lr, tr_rmse))
            print(f"  {depth:>10} {lr:>6.2f} {tr_rmse:>12.5f} {val_rmse:>12.5f} {'400':>6}")

            if _USE_WANDB:
                wandb.log({"sweep_depth": depth, "sweep_lr": lr,
                           "sweep_val_rmse": val_rmse, "sweep_train_rmse": tr_rmse})

    results.sort()
    best_val, best_depth, best_lr, best_tr = results[0]
    print(f"\n  Best from grid: max_depth={best_depth}, lr={best_lr}  "
          f"val_rmse={best_val:.5f}")

    # Phase 2: early stopping with best depth/lr
    print(f"\n  Phase 2: early stopping (n_estimators=2000, patience=40) ...")
    p2 = dict(base_params,
              max_depth=best_depth,
              learning_rate=best_lr,
              n_estimators=2000,
              early_stopping_rounds=40)
    reg2 = xgb.XGBRegressor(**p2)
    reg2.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

    best_n    = reg2.best_iteration + 1
    val_rmse2 = math.sqrt(mean_squared_error(y_vl, reg2.predict(X_vl)))
    tr_rmse2  = math.sqrt(mean_squared_error(y_tr, reg2.predict(X_tr)))
    print(f"  Early-stop best iteration: {best_n}  "
          f"train_rmse={tr_rmse2:.5f}  val_rmse={val_rmse2:.5f}")

    if _USE_WANDB:
        wandb.log({"best_n_estimators": best_n,
                   "early_stop_val_rmse": val_rmse2,
                   "best_depth": best_depth,
                   "best_lr": best_lr})

    return {"max_depth": best_depth, "learning_rate": best_lr, "n_estimators": best_n}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global _USE_WANDB

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to data folder")
    parser.add_argument("--project",   default="yelp-competition", help="wandb project name")
    parser.add_argument("--run",       default=None,               help="wandb run name")
    parser.add_argument("--ablation",  action="store_true",        help="Run ablation study")
    parser.add_argument("--no-wandb",  action="store_true",        help="Disable wandb; print metrics to console only")
    args = parser.parse_args()

    _USE_WANDB = _WANDB_AVAILABLE and not args.no_wandb
    if _USE_WANDB:
        print("[wandb] enabled")
    else:
        print("[wandb] disabled -- metrics will print to console only")

    t0 = time.time()

    # ── Load data ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading data ...")
    biz_map, usr_map, b_default, u_default = load_all(args.folder)

    print("[load] yelp_train.csv ...", end=" ", flush=True)
    train_rows = load_csv(os.path.join(args.folder, "yelp_train.csv"), has_label=True)
    print(f"{len(train_rows):,} rows")

    print("[load] yelp_val.csv ...", end=" ", flush=True)
    val_rows = load_csv(os.path.join(args.folder, "yelp_val.csv"), has_label=True)
    print(f"{len(val_rows):,} rows")

    # ── Build feature matrices ──────────────────────────────────────────────────
    print("\nBuilding feature matrices ...")
    X_train_full, y_train = build_xy(train_rows, biz_map, usr_map, b_default, u_default)
    X_val_full,   y_val   = build_xy(val_rows,   biz_map, usr_map, b_default, u_default)
    X_train_base, _       = build_xy_baseline(train_rows, biz_map, usr_map)
    X_val_base,   _       = build_xy_baseline(val_rows,   biz_map, usr_map)

    print(f"  Full feature matrix:     train {X_train_full.shape}  val {X_val_full.shape}")
    print(f"  Baseline feature matrix: train {X_train_base.shape}  val {X_val_base.shape}")

    # ── wandb init ──────────────────────────────────────────────────────────────
    if _USE_WANDB:
        wandb.init(
            project=args.project,
            name=args.run or f"feature-analysis-{int(time.time())}",
            config={
                "train_size":    len(train_rows),
                "val_size":      len(val_rows),
                "full_features": X_train_full.shape[1],
                "base_features": X_train_base.shape[1],
                **XGB_BASE_PARAMS,
            },
        )

    # ── Experiment 1: Baseline (13 features) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 1: Baseline (13 features, same as HW3 task2_2)")
    run_experiment(
        "baseline",
        X_train_base, y_train,
        X_val_base,   y_val,
    )

    # ── Experiment 2: Full (128 features) ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 2: Full (128 features)")
    reg_full, _ = run_experiment(
        "full",
        X_train_full, y_train,
        X_val_full,   y_val,
        feat_names=FEAT_NAMES,
    )

    # ── Hyperparameter tuning (grid + early stopping on full features) ─────────
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning: max_depth x learning_rate grid + early stopping")

    sweep_base = dict(XGB_BASE_PARAMS)
    sweep_base.pop("n_estimators", None)   # managed by sweep
    sweep_base.pop("max_depth",    None)
    sweep_base.pop("learning_rate", None)

    best_params = param_sweep(X_train_full, y_train, X_val_full, y_val, sweep_base)
    print(f"\n  Best params found: {best_params}")

    # Run final model with tuned params
    print("\n  Final tuned model:")
    final_params = dict(XGB_BASE_PARAMS, **best_params)
    run_experiment("tuned", X_train_full, y_train, X_val_full, y_val,
                   feat_names=FEAT_NAMES, extra_params=final_params)

    # ── Experiment 5: Full + interaction features (132 total) ─────────────────
    print("\n" + "=" * 60)
    print("Experiment 5: Full 128 features + 4 interaction terms (132 total)")

    g_avg = float(np.mean(y_train))
    print(f"  Global train avg rating: {g_avg:.4f}")

    X_tr_int = add_interactions(X_train_full, g_avg)
    X_vl_int = add_interactions(X_val_full,   g_avg)
    interact_feat_names = FEAT_NAMES + INTERACT_NAMES
    print(f"  Feature matrix: train {X_tr_int.shape}  val {X_vl_int.shape}")

    run_experiment("full_interaction", X_tr_int, y_train, X_vl_int, y_val,
                   feat_names=interact_feat_names)

    # ── Experiments 3 & 4: Top-10 features (raw + standardized) ───────────────
    print("\n" + "=" * 60)
    print("Experiment 3: Top-10 features only")

    fi       = reg_full.feature_importances_
    top10_idx   = np.argsort(fi)[::-1][:10]
    top10_names = [FEAT_NAMES[i] for i in top10_idx]
    print(f"  Selected indices: {top10_idx.tolist()}")
    print(f"  Selected features: {top10_names}")

    X_tr_t10 = X_train_full[:, top10_idx]
    X_vl_t10 = X_val_full[:,   top10_idx]

    run_experiment("top10", X_tr_t10, y_train, X_vl_t10, y_val, feat_names=top10_names)

    # ── Experiment 4: Top-10 + StandardScaler on continuous columns ────────────
    print("\n" + "=" * 60)
    print("Experiment 4: Top-10 features + StandardScaler on continuous ones")

    # Map local positions (within top-10 slice) that are continuous globally
    cont_set = set(CONTINUOUS_IDX)
    cont_local = [local_i for local_i, global_i in enumerate(top10_idx)
                  if global_i in cont_set]
    cont_feat_names = [top10_names[i] for i in cont_local]
    print(f"  Continuous features in top-10 (will be scaled): {cont_feat_names}")

    scaler = StandardScaler()
    X_tr_t10_sc = X_tr_t10.copy()
    X_vl_t10_sc = X_vl_t10.copy()
    if cont_local:
        scaler.fit(X_tr_t10[:, cont_local])
        X_tr_t10_sc[:, cont_local] = scaler.transform(X_tr_t10[:, cont_local])
        X_vl_t10_sc[:, cont_local] = scaler.transform(X_vl_t10[:, cont_local])

    run_experiment("top10_scaled", X_tr_t10_sc, y_train, X_vl_t10_sc, y_val,
                   feat_names=top10_names)

    # ── Ablation study ──────────────────────────────────────────────────────────
    if args.ablation:
        print("\n" + "=" * 60)
        print("Ablation: drop near-zero importance features")
        ablation_threshold(reg_full, X_train_full, y_train,
                           X_val_full, y_val, FEAT_NAMES, threshold=0.001)

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Total time: {time.time()-t0:.1f}s")

    if _USE_WANDB:
        print(f"wandb run: {wandb.run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
