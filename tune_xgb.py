"""
Offline XGBoost hyperparameter tuning for Yelp rating prediction.
No Spark needed — loads data directly from Python.

Usage:
    python src/tune_xgb.py <data_folder> [n_trials=30]

Outputs:
    - Top-5 hyperparameter sets ranked by val RMSE
    - Best XGBoost + CF blend weight sweep
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
_PHOTO_LABELS = ["food", "inside", "outside", "drink", "menu"]

_USE_SVD    = True
_MF_FACTORS = 10
_MF_EPOCHS  = 8

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
    """Returns {bid: list[float]} with 102 features. Categories at indices 72..101."""
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
    """Returns (tb_map, tu_map, tc_b, te_b, tc_u, tip_ub_set)."""
    tb = {}; tu = {}
    tc_b_s = {}; tc_b_n = {}; te_b_s = {}
    tc_u_s = {}; tc_u_n = {}
    tip_ub = set()
    with open(path) as f:
        for line in f:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            uid = d.get("user_id",     "")
            text = d.get("text", "") or ""
            wc  = float(len(text.split()))
            ex  = float(text.count("!"))
            tb[bid] = tb.get(bid, 0.0) + 1.0
            tu[uid] = tu.get(uid, 0.0) + 1.0
            if bid not in tc_b_s:
                tc_b_s[bid] = 0.0; tc_b_n[bid] = 0; te_b_s[bid] = 0.0
            tc_b_s[bid] += wc; tc_b_n[bid] += 1; te_b_s[bid] += ex
            if uid not in tc_u_s:
                tc_u_s[uid] = 0.0; tc_u_n[uid] = 0
            tc_u_s[uid] += wc; tc_u_n[uid] += 1
            tip_ub.add((uid, bid))
    tc_b = {b: tc_b_s[b] / tc_b_n[b] for b in tc_b_n if tc_b_n[b] > 0}
    te_b = {b: te_b_s[b] / tc_b_n[b] for b in tc_b_n if tc_b_n[b] > 0}
    tc_u = {u: tc_u_s[u] / tc_u_n[u] for u in tc_u_n if tc_u_n[u] > 0}
    return tb, tu, tc_b, te_b, tc_u, tip_ub


def load_photo(path):
    """Returns {bid: [log1p(count), food_r, inside_r, outside_r, drink_r, menu_r]}."""
    raw = {}
    with open(path) as f:
        for line in f:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            lbl = d.get("label", "")
            if bid not in raw:
                raw[bid] = [0.0] * 6
            raw[bid][0] += 1.0
            for i, pl in enumerate(_PHOTO_LABELS):
                if lbl == pl:
                    raw[bid][i + 1] += 1.0
    photo_map = {}
    _ph_def = [0.0] * 6
    for bid, v in raw.items():
        tot = v[0]
        if tot > 0:
            photo_map[bid] = [math.log1p(tot)] + [v[i + 1] / tot for i in range(5)]
        else:
            photo_map[bid] = _ph_def[:]
    return photo_map


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
# Matrix Factorization (SGD biased SVD) — identical to competition.py
# ---------------------------------------------------------------------------

def train_mf_sgd(train_rows, n_factors=10, n_epochs=8, lr=0.005, reg=0.02, seed=42):
    np.random.seed(seed)
    all_u = sorted(set(r[0] for r in train_rows))
    all_b = sorted(set(r[1] for r in train_rows))
    u2i = {u: i for i, u in enumerate(all_u)}
    b2i = {b: i for i, b in enumerate(all_b)}
    mu  = float(np.mean([float(r[2]) for r in train_rows]))
    P   = np.random.normal(0, 0.01, (len(all_u), n_factors))
    Q   = np.random.normal(0, 0.01, (len(all_b), n_factors))
    bu  = np.zeros(len(all_u))
    bi  = np.zeros(len(all_b))
    ui_arr = np.array([u2i[r[0]] for r in train_rows], dtype="int32")
    bi_arr = np.array([b2i[r[1]] for r in train_rows], dtype="int32")
    ra_arr = np.array([float(r[2]) for r in train_rows], dtype="float64")
    n   = len(ra_arr)
    bs  = 2048
    for _ in range(n_epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s: s + bs]
            uu  = ui_arr[idx]; bb = bi_arr[idx]; rr = ra_arr[idx]
            Pu  = P[uu];       Qb = Q[bb]
            err = rr - (mu + bu[uu] + bi[bb] + np.einsum("ij,ij->i", Pu, Qb))
            P[uu]  += lr * (err[:, None] * Qb - reg * Pu)
            Q[bb]  += lr * (err[:, None] * Pu - reg * Qb)
            bu[uu] += lr * (err - reg * bu[uu])
            bi[bb] += lr * (err - reg * bi[bb])
    return u2i, b2i, P, Q, bu, bi, mu


def mf_predict(uid, bid, u2i, b2i, P, Q, bu, bi, mu):
    ui   = u2i.get(uid)
    bi_i = b2i.get(bid)
    p    = mu
    if ui is not None:
        p += bu[ui]
    if bi_i is not None:
        p += bi[bi_i]
    if ui is not None and bi_i is not None:
        p += float(np.dot(P[ui], Q[bi_i]))
    return 1.0 if p < 1.0 else (5.0 if p > 5.0 else p)

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
# OOF averages + ucat (5-fold, consistent with competition.py)
# ---------------------------------------------------------------------------

def compute_oof(train_rows, biz_cats, g_avg, n_folds=5, seed=42):
    """Returns (oof_ua, oof_ia, oof_ucat, fold_ids)."""
    n   = len(train_rows)
    y   = [r[2] for r in train_rows]
    rng = random.Random(seed)
    shuf = list(range(n))
    rng.shuffle(shuf)
    fold = [0] * n
    for pos, ki in enumerate(shuf):
        fold[ki] = pos % n_folds

    oof_ua   = [g_avg] * n
    oof_ia   = [g_avg] * n
    oof_ucat = [g_avg] * n

    for k in range(n_folds):
        us = {}; uc = {}; bs = {}; bc = {}
        u_cat_sum = {}; u_cat_cnt = {}
        for i, r in enumerate(train_rows):
            if fold[i] == k:
                continue
            yi = y[i]; uid = r[0]; bid = r[1]
            us[uid] = us.get(uid, 0.0) + yi;  uc[uid] = uc.get(uid, 0) + 1
            bs[bid] = bs.get(bid, 0.0) + yi;  bc[bid] = bc.get(bid, 0) + 1
            if uid not in u_cat_sum:
                u_cat_sum[uid] = [0.0] * 30
                u_cat_cnt[uid] = [0]   * 30
            for ci in biz_cats.get(bid, []):
                u_cat_sum[uid][ci] += yi
                u_cat_cnt[uid][ci] += 1
        ua_k = {u: us[u] / uc[u] for u in us}
        ia_k = {b: bs[b] / bc[b] for b in bs}
        for i, r in enumerate(train_rows):
            if fold[i] == k:
                uid, bid = r[0], r[1]
                oof_ua[i]   = ua_k.get(uid, g_avg)
                oof_ia[i]   = ia_k.get(bid, g_avg)
                cats = biz_cats.get(bid, [])
                vals = []
                if uid in u_cat_sum:
                    for ci in cats:
                        if u_cat_cnt[uid][ci] > 0:
                            vals.append(u_cat_sum[uid][ci] / u_cat_cnt[uid][ci])
                oof_ucat[i] = sum(vals) / len(vals) if vals else g_avg

    return oof_ua, oof_ia, oof_ucat, fold

# ---------------------------------------------------------------------------
# Feature matrix building (144 features when _USE_SVD=True)
# ---------------------------------------------------------------------------

def build_features(rows, biz_map, usr_map, u_avg, i_avg, g_avg,
                   b_default, u_default, u2i, i2u, sim_cache,
                   biz_cats, u_cat_sum_all, u_cat_cnt_all, u_cats_set,
                   tip_ub_set,
                   oof_ua=None, oof_ia=None, oof_ucat=None,
                   oof_svd=None, svd_params=None):
    """
    Feature layout (144 when _USE_SVD=True):
      112 biz  : 102 base + ck + tb + 6 photo + tip_avg_wc + tip_avg_excl
      25  user : 23 base + tu + tip_avg_wc
      6   pair : OOF u_avg, OOF i_avg, diff, OOF ucat, jaccard, ub_tip_flag
      +1  SVD  : svd_score (if svd params provided)
    """
    X        = []
    cf_preds = []
    cf_ks    = []
    for idx, row in enumerate(rows):
        uid, bid = row[0], row[1]
        ua = oof_ua[idx]   if oof_ua   is not None else u_avg.get(uid, g_avg)
        ia = oof_ia[idx]   if oof_ia   is not None else i_avg.get(bid, g_avg)
        if oof_ucat is not None:
            ucat = oof_ucat[idx]
        else:
            cats = biz_cats.get(bid, [])
            vals = []
            if uid in u_cat_sum_all:
                for ci in cats:
                    if u_cat_cnt_all[uid][ci] > 0:
                        vals.append(u_cat_sum_all[uid][ci] / u_cat_cnt_all[uid][ci])
            ucat = sum(vals) / len(vals) if vals else g_avg
        uc    = u_cats_set.get(uid, set())
        bc    = set(biz_cats.get(bid, []))
        union = len(uc | bc)
        jac   = float(len(uc & bc)) / union if union > 0 else 0.0
        ubtip = 1.0 if (uid, bid) in tip_ub_set else 0.0

        feat = (list(biz_map.get(bid, b_default))
                + list(usr_map.get(uid, u_default))
                + [ua, ia, ua - ia, ucat, jac, ubtip])

        if oof_svd is not None:
            feat.append(oof_svd[idx])
        elif svd_params is not None:
            u2i_f, b2i_f, P, Q, bu, bi, mu = svd_params
            feat.append(mf_predict(uid, bid, u2i_f, b2i_f, P, Q, bu, bi, mu))

        X.append(feat)
        cf_p, cf_k = cf_predict(uid, bid, u2i, i2u, u_avg, i_avg, g_avg, sim_cache)
        cf_preds.append(cf_p)
        cf_ks.append(float(cf_k))

    X_arr  = np.array(X, dtype=np.float32)
    cp_arr = np.array(cf_preds, dtype=np.float32)
    ck_arr = np.array(cf_ks,    dtype=np.float32)
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

PARAM_GRID = {
    "max_depth":        [6, 7, 8, 9, 10],
    "learning_rate":    [0.02, 0.03, 0.05, 0.07, 0.1],
    "subsample":        [0.7, 0.75, 0.8, 0.85, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "min_child_weight": [3, 5, 7, 10, 15],
    "gamma":            [0.0, 0.05, 0.1, 0.2, 0.3],
    "reg_alpha":        [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
    "reg_lambda":       [1.0, 1.5, 2.0, 3.0, 4.0],
}


def random_search(X_tr, y_tr, X_val, y_val, n_trials=30, seed=42):
    rng      = random.Random(seed)
    y_tr_np  = np.array(y_tr, dtype=np.float32)
    y_val_np = np.array(y_val, dtype=np.float32)
    results  = []

    for trial in range(n_trials):
        params = {k: rng.choice(v) for k, v in PARAM_GRID.items()}
        model  = xgb.XGBRegressor(
            n_estimators=1500,
            early_stopping_rounds=50,
            objective="reg:squarederror",
            nthread=4,
            verbosity=0,
            seed=seed,
            **params,
        )
        model.fit(X_tr, y_tr_np, eval_set=[(X_val, y_val_np)], verbose=False)
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
# CF blend weight sweep
# ---------------------------------------------------------------------------

def sweep_blend(xgb_preds, cf_preds, cf_ks, y_val):
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

    biz_raw  = load_business(os.path.join(folder, "business.json"))
    ck_map   = load_checkin(os.path.join(folder, "checkin.json"))
    usr_raw  = load_user(os.path.join(folder, "user.json"))
    tb_map, tu_map, tc_b, te_b, tc_u, tip_ub_set = load_tip(
        os.path.join(folder, "tip.json"))
    photo_map = load_photo(os.path.join(folder, "photo.json"))

    _ph_def = [0.0] * 6
    biz_map = {bid: (f
                     + [ck_map.get(bid, 0.0), tb_map.get(bid, 0.0)]
                     + photo_map.get(bid, _ph_def)
                     + [tc_b.get(bid, 0.0), te_b.get(bid, 0.0)])
               for bid, f in biz_raw.items()}   # 112 features
    usr_map = {uid: f + [tu_map.get(uid, 0.0), tc_u.get(uid, 0.0)]
               for uid, f in usr_raw.items()}   # 25 features

    biz_vals  = list(biz_map.values())
    b_default = _col_defaults(biz_vals, len(biz_vals[0]) if biz_vals else 112)
    usr_vals  = list(usr_map.values())
    u_default = _col_defaults(usr_vals, len(usr_vals[0]) if usr_vals else 25)

    # category indices 72..101 in biz_raw (30 categories)
    biz_cats = {}
    for bid, f in biz_raw.items():
        biz_cats[bid] = [j for j in range(30) if len(f) > 72 + j and f[72 + j] == 1.0]

    train_rows = load_csv(os.path.join(folder, "yelp_train.csv"))
    val_rows   = load_csv(os.path.join(folder, "yelp_val.csv"))
    y_train    = [r[2] for r in train_rows]
    y_val      = [r[2] for r in val_rows]
    n_train    = len(train_rows)

    # CF structures from full training data
    u2i = {}; i2u = {}
    for uid, bid, rating in train_rows:
        u2i.setdefault(uid, {})[bid] = rating
        i2u.setdefault(bid, {})[uid] = rating
    u_avg = {uid: sum(d.values()) / len(d) for uid, d in u2i.items()}
    i_avg = {bid: sum(d.values()) / len(d) for bid, d in i2u.items()}
    total_r = sum(len(d) for d in i2u.values())
    g_avg   = sum(sum(d.values()) for d in i2u.values()) / total_r if total_r else 3.75

    print(f"Train={n_train}, Val={len(val_rows)}, g_avg={g_avg:.4f}")

    # Full training ucat (for val rows)
    u_cat_sum_all = {}; u_cat_cnt_all = {}; u_cats_set = {}
    for i, r in enumerate(train_rows):
        uid, bid = r[0], r[1]
        if uid not in u_cat_sum_all:
            u_cat_sum_all[uid] = [0.0] * 30
            u_cat_cnt_all[uid] = [0]   * 30
            u_cats_set[uid]    = set()
        for ci in biz_cats.get(bid, []):
            u_cat_sum_all[uid][ci] += y_train[i]
            u_cat_cnt_all[uid][ci] += 1
            u_cats_set[uid].add(ci)

    print("Computing OOF averages + ucat...")
    oof_ua, oof_ia, oof_ucat, fold = compute_oof(train_rows, biz_cats, g_avg)

    # OOF SVD
    oof_svd     = None
    svd_params  = None
    if _USE_SVD:
        print("Computing OOF SVD...")
        oof_svd = [g_avg] * n_train
        for k in range(5):
            rows_k = [train_rows[i] for i in range(n_train) if fold[i] != k]
            u2i_sv, b2i_sv, P, Q, bu_sv, bi_sv, mu_sv = train_mf_sgd(
                rows_k, _MF_FACTORS, _MF_EPOCHS, seed=42)
            for i in range(n_train):
                if fold[i] == k:
                    oof_svd[i] = mf_predict(
                        train_rows[i][0], train_rows[i][1],
                        u2i_sv, b2i_sv, P, Q, bu_sv, bi_sv, mu_sv)
        print("Training final SVD...")
        svd_params = train_mf_sgd(train_rows, _MF_FACTORS, _MF_EPOCHS + 4, seed=42)

    print("Building features (this takes ~60s for CF)...")
    sim_cache = {}
    common_kw = dict(
        biz_cats=biz_cats,
        u_cat_sum_all=u_cat_sum_all, u_cat_cnt_all=u_cat_cnt_all,
        u_cats_set=u_cats_set, tip_ub_set=tip_ub_set,
    )

    X_tr, cf_preds_tr, cf_ks_tr = build_features(
        train_rows, biz_map, usr_map, u_avg, i_avg, g_avg,
        b_default, u_default, u2i, i2u, sim_cache,
        oof_ua=oof_ua, oof_ia=oof_ia, oof_ucat=oof_ucat,
        oof_svd=oof_svd, svd_params=None, **common_kw)

    X_val, cf_preds_val, cf_ks_val = build_features(
        val_rows, biz_map, usr_map, u_avg, i_avg, g_avg,
        b_default, u_default, u2i, i2u, sim_cache,
        oof_ua=None, oof_ia=None, oof_ucat=None,
        oof_svd=None, svd_params=svd_params, **common_kw)

    print(f"Feature dim: {X_tr.shape[1]} (train), {X_val.shape[1]} (val)")
    print(f"Data + feature build: {time.time()-t0:.1f}s\n")

    # -----------------------------------------------------------------------
    # Baseline: current known-good params (n=372, tuned for 132-feat)
    # -----------------------------------------------------------------------
    print("=== Baseline (current competition.py params) ===")
    m_base = xgb.XGBRegressor(
        n_estimators=372, max_depth=8, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.6, min_child_weight=10,
        gamma=0.0, reg_alpha=0.5, reg_lambda=2.0,
        objective="reg:squarederror", nthread=4, verbosity=0, seed=42,
    )
    m_base.fit(X_tr, np.array(y_train, dtype=np.float32))
    r_base = rmse(y_val, m_base.predict(X_val))
    print(f"  n=372 (old params): RMSE={r_base:.5f}\n")

    # -----------------------------------------------------------------------
    # Find optimal n with current params (early stopping, quick)
    # -----------------------------------------------------------------------
    print("=== Finding optimal n_estimators (early stopping, current params) ===")
    m_es = xgb.XGBRegressor(
        n_estimators=1500,
        early_stopping_rounds=50,
        max_depth=8, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.6, min_child_weight=10,
        gamma=0.0, reg_alpha=0.5, reg_lambda=2.0,
        objective="reg:squarederror", nthread=4, verbosity=0, seed=42,
    )
    m_es.fit(X_tr, np.array(y_train, dtype=np.float32),
             eval_set=[(X_val, np.array(y_val, dtype=np.float32))], verbose=False)
    best_n_es = m_es.best_iteration + 1
    r_es = rmse(y_val, m_es.predict(X_val))
    print(f"  best n={best_n_es}, RMSE={r_es:.5f}\n")

    # -----------------------------------------------------------------------
    # Random search (full param sweep)
    # -----------------------------------------------------------------------
    print(f"=== Random search ({n_trials} trials) ===")
    results = random_search(X_tr, y_train, X_val, y_val, n_trials=n_trials)

    print(f"\n=== Top-5 hyperparameter sets ===")
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
    final_model.fit(X_tr, np.array(y_train, dtype=np.float32))
    xgb_preds_val = final_model.predict(X_val)

    blend_r, cw_hi, cw_lo = sweep_blend(xgb_preds_val, cf_preds_val, cf_ks_val, y_val)
    print(f"  XGB alone : RMSE={rmse(y_val, xgb_preds_val):.5f}")
    print(f"  Best blend: RMSE={blend_r:.5f}  cw_hi(>=15)={cw_hi}  cw_lo(>=5)={cw_lo}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  baseline (n=372, old params)  : RMSE={r_base:.5f}")
    print(f"  best n w/ current params      : n={best_n_es}, RMSE={r_es:.5f}")
    print(f"  best after full tuning        : n={best_n}, RMSE={best_val_r:.5f}")
    print(f"  best + CF blend               : RMSE={blend_r:.5f}")
    print(f"\nRecommended for competition.py:")
    print(f"  n_estimators: {best_n}")
    print(f"  params      : {best_params}")
    print(f"  cw_hi / cw_lo: {cw_hi} / {cw_lo}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
