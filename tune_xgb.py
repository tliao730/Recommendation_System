"""
Offline XGBoost hyperparameter tuning for Yelp rating prediction.
No Spark needed — loads data directly from Python.

Usage:
    python src/tune_xgb.py <data_folder> [n_trials=30]

Outputs:
    - Top-5 hyperparameter sets ranked by val RMSE
    - Best XGBoost + CF blend weight sweep
    - Feature ablation: base-131 vs base-131+CF-features (134)
"""

import os
import sys
import json
import math
import time
import random
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    sys.exit("pip install xgboost")

# ---------------------------------------------------------------------------
# Constants (keep in sync with competition.py)
# ---------------------------------------------------------------------------

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
_AMBIENCE = ["romantic", "intimate", "classy", "hipster", "touristy", "trendy", "upscale", "casual"]
_PARKING  = ["garage", "street", "validated", "lot", "valet"]
_MEAL     = ["dessert", "latenight", "lunch", "dinner", "breakfast", "brunch"]
_DAYS     = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_COMPS    = [
    "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute",
    "compliment_list", "compliment_note", "compliment_plain", "compliment_cool",
    "compliment_funny", "compliment_writer", "compliment_photos",
]

# ---------------------------------------------------------------------------
# Feature extraction (no Spark — pure Python file reads)
# ---------------------------------------------------------------------------

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
    if "'" + k + "': True" in s:
        return 1.0
    if "'" + k + "': False" in s:
        return 0.0
    return -1.0


def load_business(path):
    biz_map = {}
    with open(path) as f:
        for line in f:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            a   = d.get("attributes") or {}
            feat = [
                float(d.get("stars",        3.5) or 3.5),
                float(d.get("review_count",   0) or 0),
                float(d.get("is_open",        1) or 1),
                float(d.get("latitude",     0.0) or 0.0),
                float(d.get("longitude",    0.0) or 0.0),
            ]
            for b in _BOOL_ATTRS:
                v = a.get(b)
                feat.append(1.0 if v == "True" else (0.0 if v == "False" else -1.0))
            pr = a.get("RestaurantsPriceRange2")
            feat.append(float(pr) if pr in ("1", "2", "3", "4") else -1.0)
            alc = a.get("Alcohol", "")
            feat += [1.0 if alc == "none" else 0.0,
                     1.0 if alc == "beer_and_wine" else 0.0,
                     1.0 if alc == "full_bar" else 0.0]
            wifi = a.get("WiFi", "")
            feat += [1.0 if wifi == "no" else 0.0,
                     1.0 if wifi == "free" else 0.0,
                     1.0 if wifi == "paid" else 0.0]
            feat.append(_NOISE.get(str(a.get("NoiseLevel", "")), -1.0))
            feat.append(_ATTIRE.get(str(a.get("RestaurantsAttire", "")), -1.0))
            for keys, field in [(_AMBIENCE, "Ambience"),
                                 (_PARKING,  "BusinessParking"),
                                 (_MEAL,     "GoodForMeal")]:
                fs = str(a.get(field, ""))
                for k in keys:
                    feat.append(_da(fs, k))
            h = d.get("hours")
            for day in _DAYS:
                feat.append(_hours(h, day))
            state = d.get("state", "")
            for s in _TOP_STATES:
                feat.append(1.0 if state == s else 0.0)
            cats = d.get("categories") or ""
            for c in _TOP_CATS:
                feat.append(1.0 if c in cats else 0.0)
            biz_map[bid] = feat   # 102 features
    return biz_map


def load_user(path):
    usr_map = {}
    with open(path) as f:
        for line in f:
            d   = json.loads(line)
            uid = d.get("user_id", "")
            since = d.get("yelping_since", "")
            try:
                tenure = float(CURRENT_YEAR - int(since[:4]))
            except Exception:
                tenure = 10.0
            fr = d.get("friends", "")
            fc = math.log1p(len(fr.split(",")) if (fr and fr != "None") else 0)
            el     = d.get("elite", "")
            is_el  = 1.0 if (el and el != "None") else 0.0
            el_yrs = float(len(el.split(","))) if (el and el != "None") else 0.0
            us = float(d.get("useful", 0) or 0)
            fn = float(d.get("funny",  0) or 0)
            cl = float(d.get("cool",   0) or 0)
            comps = [float(d.get(k, 0) or 0) for k in _COMPS]
            feat  = [
                float(d.get("review_count",  0)  or 0),
                float(d.get("average_stars", 3.5) or 3.5),
                float(d.get("fans",          0)  or 0),
                us, fn, cl, tenure, fc, is_el, el_yrs,
            ] + comps + [sum(comps), us + fn + cl]
            usr_map[uid] = feat   # 23 features
    return usr_map


def load_checkin(path):
    ck_map = {}
    with open(path) as f:
        for line in f:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            t   = d.get("time", {})
            ck_map[bid] = float(sum(t.values())) if isinstance(t, dict) and t else 0.0
    return ck_map


def load_tip(path):
    tb_map = {}
    tu_map = {}
    with open(path) as f:
        for line in f:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            uid = d.get("user_id",     "")
            tb_map[bid] = tb_map.get(bid, 0.0) + 1.0
            tu_map[uid] = tu_map.get(uid, 0.0) + 1.0
    return tb_map, tu_map


def load_csv(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 3:
                rows.append((p[0], p[1], float(p[2])))
    return rows


def _col_defaults(rows, n_feat):
    col_sum = [0.0] * n_feat
    col_cnt = [0]   * n_feat
    for row in rows:
        for i, v in enumerate(row):
            if v != -1.0:
                col_sum[i] += v
                col_cnt[i] += 1
    return [col_sum[i] / col_cnt[i] if col_cnt[i] > 0 else 0.0
            for i in range(n_feat)]

# ---------------------------------------------------------------------------
# Item-based CF (identical to competition.py)
# ---------------------------------------------------------------------------

def _clamp(x):
    return 1.0 if x < 1.0 else (5.0 if x > 5.0 else x)


def _pearson(i, j, i2u, i_avg, cache):
    a, b = (i, j) if i < j else (j, i)
    key  = (a, b)
    if key in cache:
        return cache[key]
    ui = i2u.get(i);  uj = i2u.get(j)
    if ui is None or uj is None:
        cache[key] = 0.0
        return 0.0
    com = set(ui) & set(uj)
    if len(com) < 2:
        cache[key] = 0.0
        return 0.0
    ai  = i_avg.get(i, 0.0)
    aj  = i_avg.get(j, 0.0)
    num = di2 = dj2 = 0.0
    for u in com:
        di  = ui[u] - ai;  dj  = uj[u] - aj
        num += di * dj;    di2 += di * di;  dj2 += dj * dj
    denom = math.sqrt(di2 * dj2)
    sim   = (num / denom if denom > 0.0 else 0.0) * (len(com) / (len(com) + 2.0))
    cache[key] = sim
    return sim


def cf_predict(uid, bid, u2i, i2u, u_avg, i_avg, g_avg, cache, top_n=30):
    """Returns (prediction, neighbor_count)."""
    has_u = uid in u2i
    has_i = bid in i2u
    if not has_u and not has_i:
        return _clamp(g_avg), 0
    if not has_u:
        return _clamp(i_avg.get(bid, g_avg)), 0
    if not has_i:
        return _clamp(u_avg.get(uid, g_avg)), 0
    rated = u2i[uid]
    sims  = []
    for nb in rated:
        if nb == bid:
            continue
        s = _pearson(bid, nb, i2u, i_avg, cache)
        if s > 0.0:
            sims.append((nb, s))
    if not sims:
        base = 0.15 * u_avg.get(uid, g_avg) + 0.85 * i_avg.get(bid, g_avg)
        return _clamp(base), 0
    sims.sort(key=lambda x: x[1], reverse=True)
    top   = sims[:top_n]
    b_ui  = 0.15 * u_avg.get(uid, g_avg) + 0.85 * i_avg.get(bid, g_avg)
    num   = denom = 0.0
    for nb, s in top:
        b_un  = 0.15 * u_avg.get(uid, g_avg) + 0.85 * i_avg.get(nb, g_avg)
        num   += s * (float(rated[nb]) - b_un)
        denom += s
    w   = denom / (denom + 3.0)
    cf  = b_ui + num / denom if denom > 0.0 else b_ui
    return _clamp(w * cf + (1.0 - w) * b_ui), len(top)

# ---------------------------------------------------------------------------
# OOF averages
# ---------------------------------------------------------------------------

def compute_oof_avgs(train_rows, g_avg, n_folds=5, seed=42):
    n = len(train_rows)
    y = [r[2] for r in train_rows]
    rng = random.Random(seed)
    shuf = list(range(n))
    rng.shuffle(shuf)
    fold = [0] * n
    for pos, ki in enumerate(shuf):
        fold[ki] = pos % n_folds

    oof_ua = [g_avg] * n
    oof_ia = [g_avg] * n
    for k in range(n_folds):
        us = {}; uc = {}; bs = {}; bc = {}
        for i, r in enumerate(train_rows):
            if fold[i] == k:
                continue
            yi = y[i]; uid = r[0]; bid = r[1]
            us[uid] = us.get(uid, 0.0) + yi;  uc[uid] = uc.get(uid, 0) + 1
            bs[bid] = bs.get(bid, 0.0) + yi;  bc[bid] = bc.get(bid, 0) + 1
        ua_k = {u: us[u] / uc[u] for u in us}
        ia_k = {b: bs[b] / bc[b] for b in bs}
        for i, r in enumerate(train_rows):
            if fold[i] == k:
                oof_ua[i] = ua_k.get(r[0], g_avg)
                oof_ia[i] = ia_k.get(r[1], g_avg)
    return oof_ua, oof_ia

# ---------------------------------------------------------------------------
# Feature matrix building
# ---------------------------------------------------------------------------

def build_features(rows, biz_map, usr_map, u_avg, i_avg, g_avg,
                   b_default, u_default, u2i, i2u, sim_cache,
                   oof_ua=None, oof_ia=None, add_cf=True):
    """
    Returns X (np.ndarray, float32), cf_preds (array), cf_ks (array).

    oof_ua / oof_ia: if provided (same length as rows), used as the
    u_avg/i_avg features instead of the full-data averages.  Pass None
    for val/test rows (which use full training averages — no leakage).

    add_cf: if True, append cf_score, cf_k, is_cold as extra features.
    CF is always computed from the full training u2i/i2u; val is clean
    because it was never in the training structures.
    Training CF has minor self-inclusion leakage, acceptable for tuning.
    """
    X         = []
    cf_preds  = []
    cf_ks     = []
    for idx, row in enumerate(rows):
        uid = row[0];  bid = row[1]
        ua  = oof_ua[idx] if oof_ua is not None else u_avg.get(uid, g_avg)
        ia  = oof_ia[idx] if oof_ia is not None else i_avg.get(bid, g_avg)
        feat = (list(biz_map.get(bid, b_default))
                + list(usr_map.get(uid, u_default))
                + [ua, ia, ua - ia])
        X.append(feat)
        cf_p, cf_k = cf_predict(uid, bid, u2i, i2u, u_avg, i_avg, g_avg, sim_cache)
        cf_preds.append(cf_p)
        cf_ks.append(float(cf_k))

    X_arr  = np.array(X, dtype=np.float32)
    cp_arr = np.array(cf_preds, dtype=np.float32)
    ck_arr = np.array(cf_ks,    dtype=np.float32)

    if add_cf:
        is_cold = (ck_arr < 5).astype(np.float32)
        X_arr = np.hstack([X_arr,
                           cp_arr.reshape(-1, 1),
                           ck_arr.reshape(-1, 1),
                           is_cold.reshape(-1, 1)])
    return X_arr, cp_arr, ck_arr

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))

# ---------------------------------------------------------------------------
# Hyperparameter random search
# ---------------------------------------------------------------------------

# Search space — comment out a line to exclude it from the sweep.
PARAM_GRID = {
    "max_depth":        [6, 7, 8, 9, 10],
    "learning_rate":    [0.02, 0.03, 0.05, 0.07, 0.1],
    "subsample":        [0.7, 0.75, 0.8, 0.85, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [3, 5, 7, 10],
    "gamma":            [0.0, 0.05, 0.1, 0.2, 0.3],
    "reg_alpha":        [0.0, 0.05, 0.1, 0.2, 0.5],
    "reg_lambda":       [1.0, 1.5, 2.0, 3.0, 4.0],
}


def random_search(X_tr, y_tr, X_val, y_val, n_trials=30, seed=42):
    """
    For each trial: sample random params, train with early stopping on val set,
    record (val_rmse, best_n_trees, params).  Returns list sorted by val_rmse.
    """
    rng     = random.Random(seed)
    y_tr_np = np.array(y_tr, dtype=np.float32)
    y_val_np = np.array(y_val, dtype=np.float32)
    results  = []

    for trial in range(n_trials):
        params = {k: rng.choice(v) for k, v in PARAM_GRID.items()}

        model = xgb.XGBRegressor(
            n_estimators=1500,
            early_stopping_rounds=50,
            objective="reg:squarederror",   # local Python 3.13+
            nthread=4,
            verbosity=0,
            seed=seed,
            **params,
        )
        model.fit(
            X_tr, y_tr_np,
            eval_set=[(X_val, y_val_np)],
            verbose=False,
        )
        best_n = model.best_iteration + 1
        preds  = model.predict(X_val)
        val_r  = rmse(y_val_np, preds)
        results.append((val_r, best_n, params))

        print(
            f"  trial {trial+1:3d}/{n_trials} | RMSE={val_r:.5f} | "
            f"n={best_n:4d} | depth={params['max_depth']} "
            f"lr={params['learning_rate']} sub={params['subsample']}"
        )

    results.sort(key=lambda x: x[0])
    return results

# ---------------------------------------------------------------------------
# CF blend weight sweep (post-hoc, over val predictions)
# ---------------------------------------------------------------------------

def sweep_blend(xgb_preds, cf_preds, cf_ks, y_val):
    """Grid search over cw_hi (>=15 nbrs) and cw_lo (>=5 nbrs)."""
    cw_options = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    best = (float("inf"), 0.0, 0.0)
    for cw_hi in cw_options:
        for cw_lo in cw_options:
            if cw_lo > cw_hi:
                continue
            preds = []
            for xp, cp, ck in zip(xgb_preds, cf_preds, cf_ks):
                if ck >= 15:
                    cw = cw_hi
                elif ck >= 5:
                    cw = cw_lo
                else:
                    cw = 0.0
                preds.append(_clamp(cw * float(cp) + (1.0 - cw) * float(xp)))
            r = rmse(y_val, preds)
            if r < best[0]:
                best = (r, cw_hi, cw_lo)
    return best

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python src/tune_xgb.py <data_folder> [n_trials=30]")

    folder   = sys.argv[1]
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    t0 = time.time()
    print("=== Loading data ===")

    biz_raw       = load_business(os.path.join(folder, "business.json"))
    ck_map        = load_checkin(os.path.join(folder, "checkin.json"))
    tb_map, tu_map = load_tip(os.path.join(folder, "tip.json"))
    usr_raw       = load_user(os.path.join(folder, "user.json"))

    biz_map = {bid: f + [ck_map.get(bid, 0.0), tb_map.get(bid, 0.0)]
               for bid, f in biz_raw.items()}   # 104 features
    usr_map = {uid: f + [tu_map.get(uid, 0.0)]
               for uid, f in usr_raw.items()}   # 24 features

    biz_vals  = list(biz_map.values())
    b_default = _col_defaults(biz_vals, len(biz_vals[0]) if biz_vals else 104)
    usr_vals  = list(usr_map.values())
    u_default = _col_defaults(usr_vals, len(usr_vals[0]) if usr_vals else 24)

    train_rows = load_csv(os.path.join(folder, "yelp_train.csv"))
    val_rows   = load_csv(os.path.join(folder, "yelp_val.csv"))
    y_train    = [r[2] for r in train_rows]
    y_val      = [r[2] for r in val_rows]

    # CF structures from full training data
    u2i = {}; i2u = {}
    for uid, bid, rating in train_rows:
        u2i.setdefault(uid, {})[bid] = rating
        i2u.setdefault(bid, {})[uid] = rating
    u_avg = {uid: sum(d.values()) / len(d) for uid, d in u2i.items()}
    i_avg = {bid: sum(d.values()) / len(d) for bid, d in i2u.items()}
    total_r = sum(len(d) for d in i2u.values())
    g_avg   = sum(sum(d.values()) for d in i2u.values()) / total_r if total_r else 3.75

    print(f"Train={len(train_rows)}, Val={len(val_rows)}, g_avg={g_avg:.4f}")

    # OOF u_avg / i_avg for training rows (no leakage)
    print("Computing OOF averages...")
    oof_ua, oof_ia = compute_oof_avgs(train_rows, g_avg)

    # -----------------------------------------------------------------------
    # Feature matrices — two variants:
    #   A) base-131  (104 biz + 24 usr + OOF u_avg + OOF i_avg + diff)
    #   B) base-134  (A + cf_score + cf_k + is_cold)
    # -----------------------------------------------------------------------
    print("Building features (this takes ~60s for CF)...")
    sim_cache = {}

    X_tr_full, cf_preds_tr, cf_ks_tr = build_features(
        train_rows, biz_map, usr_map, u_avg, i_avg, g_avg,
        b_default, u_default, u2i, i2u, sim_cache,
        oof_ua=oof_ua, oof_ia=oof_ia, add_cf=True,
    )
    X_val_full, cf_preds_val, cf_ks_val = build_features(
        val_rows, biz_map, usr_map, u_avg, i_avg, g_avg,
        b_default, u_default, u2i, i2u, sim_cache,
        oof_ua=None, oof_ia=None, add_cf=True,
    )

    # Base-131 slices (drop last 3 CF columns)
    X_tr_base  = X_tr_full[:, :-3]
    X_val_base = X_val_full[:, :-3]

    print(f"Feature dims: base={X_tr_base.shape[1]}, full={X_tr_full.shape[1]}")
    print(f"Feature build: {time.time()-t0:.1f}s\n")

    # -----------------------------------------------------------------------
    # Quick baseline check with current known-good params
    # -----------------------------------------------------------------------
    def quick_eval(X_tr, X_val, label):
        m = xgb.XGBRegressor(
            n_estimators=474, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=2.0,
            objective="reg:squarederror", nthread=4, verbosity=0, seed=42,
        )
        m.fit(X_tr, np.array(y_train, dtype=np.float32))
        r = rmse(y_val, m.predict(X_val))
        print(f"  baseline ({label}) RMSE={r:.5f}")
        return m, r

    print("=== Baseline (n=474, known-good params) ===")
    m_base, r_base = quick_eval(X_tr_base,  X_val_base,  "base-131")
    m_full, r_full = quick_eval(X_tr_full,  X_val_full,  "base+CF-134")
    print()

    # -----------------------------------------------------------------------
    # Random search — tune on whichever feature set is better
    # -----------------------------------------------------------------------
    use_full = r_full <= r_base
    X_tr_tune  = X_tr_full  if use_full else X_tr_base
    X_val_tune = X_val_full if use_full else X_val_base
    feat_label = "base+CF-134" if use_full else "base-131"
    print(f"=== Random search on {feat_label} ({n_trials} trials) ===")
    results = random_search(X_tr_tune, y_train, X_val_tune, y_val, n_trials=n_trials)

    print(f"\n=== Top-5 hyperparameter sets ({feat_label}) ===")
    for rank, (val_r, best_n, params) in enumerate(results[:5], 1):
        print(f"  #{rank}  RMSE={val_r:.5f}  n_trees={best_n}")
        print(f"       {params}")

    # -----------------------------------------------------------------------
    # CF blend weight sweep using best-found params
    # -----------------------------------------------------------------------
    best_val_r, best_n, best_params = results[0]
    print(f"\n=== CF blend sweep (best params, n={best_n}) ===")
    final_model = xgb.XGBRegressor(
        n_estimators=best_n,
        objective="reg:squarederror",
        nthread=4, verbosity=0, seed=42,
        **best_params,
    )
    final_model.fit(X_tr_tune, np.array(y_train, dtype=np.float32))
    xgb_preds_val = final_model.predict(X_val_tune)

    blend_r, cw_hi, cw_lo = sweep_blend(xgb_preds_val, cf_preds_val, cf_ks_val, y_val)
    print(f"  XGB alone : RMSE={rmse(y_val, xgb_preds_val):.5f}")
    print(f"  Best blend: RMSE={blend_r:.5f}  cw_hi(>=15 nbrs)={cw_hi}  cw_lo(>=5 nbrs)={cw_lo}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  baseline base-131   : RMSE={r_base:.5f}")
    print(f"  baseline base+CF-134: RMSE={r_full:.5f}")
    print(f"  best after tuning   : RMSE={best_val_r:.5f}")
    print(f"  best + CF blend     : RMSE={blend_r:.5f}")
    print(f"\nRecommended for competition.py:")
    print(f"  feature set : {feat_label}")
    print(f"  n_estimators: {best_n}")
    print(f"  params      : {best_params}")
    print(f"  cw_hi / cw_lo: {cw_hi} / {cw_lo}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
