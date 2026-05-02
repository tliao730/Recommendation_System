import os
import sys
import time
import json
import math
import numpy as np
import xgboost as xgb
from pyspark import SparkContext

# -- Module-level constants (must be serialisable to Spark workers) ------------

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

_NOISE  = {"quiet": 0.0, "average": 1.0, "loud": 2.0, "very_loud": 3.0}
_ATTIRE = {"casual": 0.0, "dressy": 1.0, "formal": 2.0}
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

# Features dropped for importance ≤ 0.004 (measured by XGBoost weight importance).
# Indices are in the full 144-feature vector (when _USE_SVD=True).
# Dropping 25 features: 144 → 119. XGBoost trains ~17% faster per tree.
_DROP_FEAT_IDX = frozenset([
    2,                       # b_is_open (always 1, zero importance)
    8, 11, 12, 13, 22, 23,  # CoatCheck, GoodForDancing, GoodForKids, HappyHour, WheelchairAccessible, BYOB
    28,                      # alc_none
    31, 32,                  # wifi_no, wifi_free
    47, 48,                  # park_lot, park_valet
    52,                      # meal_dinner
    62, 69,                  # state_AZ, state_AB (zero importance)
    74, 80, 94,              # cat_Food, cat_Nightlife, cat_Burgers
    105, 108,                # photo_food_r, photo_drink_r
    123, 124, 125, 126,      # compliment_more, compliment_profile, compliment_cute, compliment_list
    141,                     # jaccard (correlated with oof_ucat, barely used)
])

# -- Feature Extraction (RDD map functions) ------------------------------------

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
    """Dict-attr string search: return 1/0/-1 for True/False/missing."""
    if "'" + k + "': True" in s:
        return 1.0
    if "'" + k + "': False" in s:
        return 0.0
    return -1.0


def extract_business(line):
    """
    RDD map -> (business_id, list[float]) with 102 features.
    Layout: [stars, review_count, is_open, lat, lon,
             22xbool_attr, price_range, 3xalcohol, 3xwifi, noise, attire,
             8xambience, 5xparking, 6xmeal, 7xhours, 10xstate, 30xcategory]
    Categories are at indices 72..101 (used by biz_cats precomputation).
    """
    d   = json.loads(line)
    a   = d.get("attributes") or {}
    bid = d.get("business_id", "")

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

    alc = a.get("Alcohol", "")
    f += [1.0 if alc == "none" else 0.0,
          1.0 if alc == "beer_and_wine" else 0.0,
          1.0 if alc == "full_bar" else 0.0]

    wifi = a.get("WiFi", "")
    f += [1.0 if wifi == "no" else 0.0,
          1.0 if wifi == "free" else 0.0,
          1.0 if wifi == "paid" else 0.0]

    f.append(_NOISE.get(str(a.get("NoiseLevel", "")), -1.0))
    f.append(_ATTIRE.get(str(a.get("RestaurantsAttire", "")), -1.0))

    for keys, field in [(_AMBIENCE, "Ambience"),
                        (_PARKING,  "BusinessParking"),
                        (_MEAL,     "GoodForMeal")]:
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

    return bid, f  # 102 features


def extract_user(line):
    """
    RDD map: (user_id, list[float]) with 23 features.
    Layout: [review_count, avg_stars, fans, useful, funny, cool, tenure,
             friend_log, is_elite, elite_yrs, 11xcompliment, total_comp, engagement]
    """
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

    us  = float(d.get("useful", 0) or 0)
    fn  = float(d.get("funny",  0) or 0)
    cl  = float(d.get("cool",   0) or 0)

    comps = [float(d.get(k, 0) or 0) for k in _COMPS]

    f = [
        float(d.get("review_count",   0)   or 0),
        float(d.get("average_stars", 3.5)  or 3.5),
        float(d.get("fans",           0)   or 0),
        us, fn, cl, tenure, fc, is_el, el_yrs,
    ] + comps + [sum(comps), us + fn + cl]

    return uid, f  # 23 features


def extract_checkin(line):
    """RDD map: (business_id, total_checkin_count)."""
    d   = json.loads(line)
    bid = d.get("business_id", "")
    t   = d.get("time", {})
    cnt = float(sum(t.values())) if isinstance(t, dict) and t else 0.0
    return bid, cnt


def extract_tip(line):
    """
    RDD flatMap: [(b_<bid>, 1.0), (u_<uid>, 1.0)]
    Single pass produces both business-tip and user-tip counts.
    """
    d = json.loads(line)
    return [
        ("b_" + d.get("business_id", ""), 1.0),
        ("u_" + d.get("user_id",      ""), 1.0),
    ]


def extract_tip_content(line):
    """
    RDD flatMap: [(b_<bid>, [1, words, excl]), (u_<uid>, [1, words, excl])]
    Extracts avg word count and exclamation mark count per tip.
    """
    d    = json.loads(line)
    text = d.get("text", "") or ""
    wc   = float(len(text.split()))
    ex   = float(text.count("!"))
    return [
        ("b_" + d.get("business_id", ""), [1.0, wc, ex]),
        ("u_" + d.get("user_id",      ""), [1.0, wc, ex]),
    ]


def _add_vec3(a, b):
    """Element-wise sum of two 3-element lists (reduceByKey combiner)."""
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _tip_ub_pair(line):
    """RDD map: (user_id, business_id) for implicit feedback set."""
    d = json.loads(line)
    return (d.get("user_id", ""), d.get("business_id", ""))


def extract_photo(line):
    """
    RDD map: (business_id, [1, is_food, is_inside, is_outside, is_drink, is_menu])
    reduceByKey sums counts; then normalize to get percentages.
    """
    d   = json.loads(line)
    bid = d.get("business_id", "")
    lbl = d.get("label", "")
    v   = [1.0] + [1.0 if lbl == pl else 0.0 for pl in _PHOTO_LABELS]
    return bid, v


def _photo_add(a, b):
    """Element-wise sum for 6-element photo count vectors."""
    return [a[i] + b[i] for i in range(6)]


# -- Helper: column-wise default (mean, excluding -1 sentinel) -----------------

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


# -- Item-Based CF -------------------------------------------------------------

def _clamp(x):
    return 1.0 if x < 1.0 else (5.0 if x > 5.0 else x)


def _pearson(i, j, i2u, i_avg, cache):
    a, b = (i, j) if i < j else (j, i)
    key  = (a, b)
    if key in cache:
        return cache[key]

    ui = i2u.get(i)
    uj = i2u.get(j)
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
        di  = ui[u] - ai
        dj  = uj[u] - aj
        num += di * dj
        di2 += di * di
        dj2 += dj * dj

    denom = math.sqrt(di2 * dj2)
    sim   = (num / denom if denom > 0.0 else 0.0) * (len(com) / (len(com) + 2.0))
    cache[key] = sim
    return sim


def cf_predict(uid, bid, u2i, i2u, u_avg, i_avg, g_avg, cache, top_n=30):
    """Baseline-adjusted item-based CF. Returns (prediction, neighbor_count)."""
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
    top = sims[:top_n]

    b_ui  = 0.15 * u_avg.get(uid, g_avg) + 0.85 * i_avg.get(bid, g_avg)
    num   = denom = 0.0
    for nb, s in top:
        b_un  = 0.15 * u_avg.get(uid, g_avg) + 0.85 * i_avg.get(nb, g_avg)
        num   += s * (float(rated[nb]) - b_un)
        denom += s

    w  = denom / (denom + 3.0)
    cf = b_ui + num / denom if denom > 0.0 else b_ui
    return _clamp(w * cf + (1.0 - w) * b_ui), len(top)


# -- Matrix Factorization (SGD biased SVD) ------------------------------------

def train_mf_sgd(train_rows, n_factors=50, n_epochs=20, lr=0.005, reg=0.02, seed=42):
    """
    Biased SVD via mini-batch SGD.
    pred(u,i) = mu + b_u + b_i + P_u . Q_i
    Returns (u2i, b2i, P, Q, bu, bi, mu).
    """
    np.random.seed(seed)

    all_u = sorted(set(r[0] for r in train_rows))
    all_b = sorted(set(r[1] for r in train_rows))
    u2i = {u: i for i, u in enumerate(all_u)}
    b2i = {b: i for i, b in enumerate(all_b)}
    n_u, n_b = len(all_u), len(all_b)

    mu = float(np.mean([float(r[2]) for r in train_rows]))

    P  = np.random.normal(0, 0.01, (n_u, n_factors))
    Q  = np.random.normal(0, 0.01, (n_b, n_factors))
    bu = np.zeros(n_u)
    bi = np.zeros(n_b)

    ui_arr = np.array([u2i[r[0]] for r in train_rows], dtype="int32")
    bi_arr = np.array([b2i[r[1]] for r in train_rows], dtype="int32")
    ra_arr = np.array([float(r[2]) for r in train_rows], dtype="float64")
    n = len(ra_arr)

    bs = 2048
    for _ in range(n_epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s: s + bs]
            uu  = ui_arr[idx]
            bb  = bi_arr[idx]
            rr  = ra_arr[idx]
            Pu  = P[uu]
            Qb  = Q[bb]
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
    return _clamp(p)


# ---------------------------------------------------------------------------
# Tuning flags — change here to trade accuracy vs Vocareum time budget
# ---------------------------------------------------------------------------

# OOF CF as feature: leakage through shared similarity cache → RMSE 1.097.
# Keep False; CF runs as post-hoc blend only.
_USE_OOF_CF = False

# SVD feature: OOF 5-fold + 1 final train.  Also exposes P_u and Q_i vectors.
# Estimated Vocareum time: +5-7 min.
_USE_SVD    = True
_MF_FACTORS = 10
_MF_EPOCHS  = 8   # per OOF fold; final model uses _MF_EPOCHS + 4

# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    folder_path = sys.argv[1]
    test_file   = sys.argv[2]
    output_file = sys.argv[3]

    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    sc = SparkContext("local[*]", "competition")
    sc.setLogLevel("ERROR")

    biz_path   = os.path.join(folder_path, "business.json")
    usr_path   = os.path.join(folder_path, "user.json")
    ck_path    = os.path.join(folder_path, "checkin.json")
    tip_path   = os.path.join(folder_path, "tip.json")
    photo_path = os.path.join(folder_path, "photo.json")
    train_path = os.path.join(folder_path, "yelp_train.csv")

    # -- Step 1: Load & broadcast auxiliary data via RDD ----------------------

    biz_raw = sc.textFile(biz_path).map(extract_business).collectAsMap()

    ck_map = (sc.textFile(ck_path)
                .map(extract_checkin)
                .reduceByKey(lambda a, b: a + b)
                .collectAsMap())

    # Tip: count aggregates + content stats + implicit (uid,bid) pairs
    tip_rdd = sc.textFile(tip_path).persist()

    tip_all = (tip_rdd
                 .flatMap(extract_tip)
                 .reduceByKey(lambda a, b: a + b)
                 .collectAsMap())
    tb_map = {k[2:]: v for k, v in tip_all.items() if k[:2] == "b_"}
    tu_map = {k[2:]: v for k, v in tip_all.items() if k[:2] == "u_"}

    tip_content = (tip_rdd
                     .flatMap(extract_tip_content)
                     .reduceByKey(_add_vec3)
                     .collectAsMap())
    # avg words per tip per business/user; avg exclamations per tip per business
    tc_b = {k[2:]: v[1] / v[0] for k, v in tip_content.items() if k[:2] == "b_" and v[0] > 0}
    te_b = {k[2:]: v[2] / v[0] for k, v in tip_content.items() if k[:2] == "b_" and v[0] > 0}
    tc_u = {k[2:]: v[1] / v[0] for k, v in tip_content.items() if k[:2] == "u_" and v[0] > 0}

    # user-business pairs from tip.json (implicit positive feedback)
    tip_ub_set = set(tip_rdd.map(_tip_ub_pair).distinct().collect())

    tip_rdd.unpersist()

    # Photo: log1p(count) + label distribution (food/inside/outside/drink/menu)
    photo_raw = (sc.textFile(photo_path)
                   .map(extract_photo)
                   .reduceByKey(_photo_add)
                   .collectAsMap())
    _ph_def = [0.0] * 6
    photo_map = {}
    for _bid, _v in photo_raw.items():
        _tot = _v[0]
        if _tot > 0:
            photo_map[_bid] = [math.log1p(_tot)] + [_v[i + 1] / _tot for i in range(5)]
        else:
            photo_map[_bid] = _ph_def[:]

    usr_raw = sc.textFile(usr_path).map(extract_user).collectAsMap()

    # biz_map: 102 base + ck + tb + 6 photo + 2 tip_content = 112 features
    biz_map = {bid: (list(f)
                     + [ck_map.get(bid, 0.0), tb_map.get(bid, 0.0)]
                     + photo_map.get(bid, _ph_def)
                     + [tc_b.get(bid, 0.0), te_b.get(bid, 0.0)])
               for bid, f in biz_raw.items()}

    # usr_map: 23 base + tu + tc_u = 25 features
    usr_map = {uid: list(f) + [tu_map.get(uid, 0.0), tc_u.get(uid, 0.0)]
               for uid, f in usr_raw.items()}

    biz_vals  = list(biz_map.values())
    n_bfeat   = len(biz_vals[0]) if biz_vals else 112
    b_default = _col_defaults(biz_vals, n_bfeat)

    usr_vals  = list(usr_map.values())
    n_ufeat   = len(usr_vals[0]) if usr_vals else 25
    u_default = _col_defaults(usr_vals, n_ufeat)

    # Precompute biz → category indices (categories are at positions 72..101 in extract_business)
    biz_cats = {}
    for bid, f in biz_raw.items():
        biz_cats[bid] = [j for j in range(30) if len(f) > 72 + j and f[72 + j] == 1.0]

    # -- Step 2: Load & persist train RDD, build CF structures ----------------

    train_raw = sc.textFile(train_path)
    hdr       = train_raw.first()
    train_rdd = (train_raw
                 .filter(lambda x: x != hdr)
                 .map(lambda s: s.split(","))
                 .persist())

    user_data = (train_rdd
                 .map(lambda r: (r[0], (r[1], float(r[2]))))
                 .groupByKey()
                 .mapValues(lambda ps: {p[0]: p[1] for p in ps})
                 .collectAsMap())
    u2i   = user_data
    u_avg = {uid: sum(d.values()) / len(d) for uid, d in user_data.items()}

    item_data = (train_rdd
                 .map(lambda r: (r[1], (r[0], float(r[2]))))
                 .groupByKey()
                 .mapValues(lambda ps: {p[0]: p[1] for p in ps})
                 .collectAsMap())
    i2u   = item_data
    i_avg = {bid: sum(d.values()) / len(d) for bid, d in item_data.items()}

    total_r = sum(len(d) for d in item_data.values())
    g_avg   = sum(sum(d.values()) for d in item_data.values()) / total_r if total_r else 3.75

    sim_cache = {}

    # -- Step 3: Load test pairs early (sim_cache warms before test eval) -----

    test_raw   = sc.textFile(test_file)
    test_hdr   = test_raw.first()
    test_pairs = (test_raw
                  .filter(lambda x: x != test_hdr)
                  .map(lambda s: s.split(","))
                  .map(lambda p: (p[0], p[1]))
                  .collect())

    print("Data loaded: {:.1f}s".format(time.time() - t0))

    # -- Step 4: Collect train rows and compute OOF features ------------------

    train_rows = train_rdd.collect()
    y_train    = [float(r[2]) for r in train_rows]
    n_train    = len(train_rows)

    import random as _rnd
    _rnd.seed(42)
    _shuf = list(range(n_train))
    _rnd.shuffle(_shuf)
    _fold = [0] * n_train
    for _pos, _ki in enumerate(_shuf):
        _fold[_ki] = _pos % 5

    # -- 4a. OOF u_avg / i_avg + user-category preference (single 5-fold pass) -

    oof_ua   = [g_avg] * n_train
    oof_ia   = [g_avg] * n_train
    oof_ucat = [g_avg] * n_train   # user's avg rating for this biz's categories

    for _k in range(5):
        _us = {}
        _uc = {}
        _bs = {}
        _bc = {}
        _u_cat_sum = {}
        _u_cat_cnt = {}
        for _i, r in enumerate(train_rows):
            if _fold[_i] == _k:
                continue
            _y = y_train[_i]
            _uid = r[0]
            _bid = r[1]
            _us[_uid] = _us.get(_uid, 0.0) + _y
            _uc[_uid] = _uc.get(_uid, 0) + 1
            _bs[_bid] = _bs.get(_bid, 0.0) + _y
            _bc[_bid] = _bc.get(_bid, 0) + 1
            if _uid not in _u_cat_sum:
                _u_cat_sum[_uid] = [0.0] * 30
                _u_cat_cnt[_uid] = [0]   * 30
            for _ci in biz_cats.get(_bid, []):
                _u_cat_sum[_uid][_ci] += _y
                _u_cat_cnt[_uid][_ci] += 1
        _ua = {u: _us[u] / _uc[u] for u in _us}
        _ia = {b: _bs[b] / _bc[b] for b in _bs}
        for _i, r in enumerate(train_rows):
            if _fold[_i] == _k:
                oof_ua[_i] = _ua.get(r[0], g_avg)
                oof_ia[_i] = _ia.get(r[1], g_avg)
                _uid = r[0]
                _bid = r[1]
                _cats = biz_cats.get(_bid, [])
                _vals = []
                if _uid in _u_cat_sum:
                    for _ci in _cats:
                        if _u_cat_cnt[_uid][_ci] > 0:
                            _vals.append(_u_cat_sum[_uid][_ci] / _u_cat_cnt[_uid][_ci])
                oof_ucat[_i] = sum(_vals) / len(_vals) if _vals else g_avg

    # -- 4b. Jaccard category similarity (no rating leakage) ------------------
    # Measures: fraction of the business's categories the user has historically visited.

    u_cats_set = {}
    for r in train_rows:
        _uid, _bid = r[0], r[1]
        if _uid not in u_cats_set:
            u_cats_set[_uid] = set()
        u_cats_set[_uid].update(biz_cats.get(_bid, []))

    jaccard_train = []
    for r in train_rows:
        _uid, _bid = r[0], r[1]
        _uc = u_cats_set.get(_uid, set())
        _bc = set(biz_cats.get(_bid, []))
        _union = len(_uc | _bc)
        jaccard_train.append(float(len(_uc & _bc)) / _union if _union > 0 else 0.0)

    # -- 4c. Full u_cat_avg (for test-time user-category feature) -------------

    u_cat_sum_all = {}
    u_cat_cnt_all = {}
    for _i, r in enumerate(train_rows):
        _uid, _bid = r[0], r[1]
        if _uid not in u_cat_sum_all:
            u_cat_sum_all[_uid] = [0.0] * 30
            u_cat_cnt_all[_uid] = [0]   * 30
        for _ci in biz_cats.get(_bid, []):
            u_cat_sum_all[_uid][_ci] += y_train[_i]
            u_cat_cnt_all[_uid][_ci] += 1

    # -- 4d. OOF CF (approximate, disabled by default) ------------------------

    oof_cf   = [g_avg] * n_train
    oof_cf_k = [0]     * n_train

    if _USE_OOF_CF:
        print("OOF CF start: {:.1f}s".format(time.time() - t0))
        for _k in range(5):
            _us = {}
            _uc = {}
            _bs = {}
            _bc = {}
            for _i, r in enumerate(train_rows):
                if _fold[_i] == _k:
                    continue
                _y = y_train[_i]
                _uid = r[0]
                _bid = r[1]
                _us[_uid] = _us.get(_uid, 0.0) + _y
                _uc[_uid] = _uc.get(_uid, 0) + 1
                _bs[_bid] = _bs.get(_bid, 0.0) + _y
                _bc[_bid] = _bc.get(_bid, 0) + 1
            _ua_k = {u: _us[u] / _uc[u] for u in _us}
            _ia_k = {b: _bs[b] / _bc[b] for b in _bs}
            for _i, r in enumerate(train_rows):
                if _fold[_i] == _k:
                    _p, _nk = cf_predict(r[0], r[1], u2i, i2u,
                                         _ua_k, _ia_k, g_avg, sim_cache)
                    oof_cf[_i]   = _p
                    oof_cf_k[_i] = _nk
            print("  OOF CF fold {}: {:.1f}s".format(_k, time.time() - t0))

    # -- 4e. OOF SVD + latent vectors (P_u, Q_i per fold) --------------------

    oof_svd = [g_avg] * n_train
    _u2i_final = _b2i_final = _P_f = _Q_f = _bu_f = _bi_f = _mu_f = None

    if _USE_SVD:
        print("OOF SVD start: {:.1f}s".format(time.time() - t0))
        for _k in range(5):
            _rows_k = [train_rows[_i] for _i in range(n_train) if _fold[_i] != _k]
            _u2i_sv, _b2i_sv, _P, _Q, _bu_sv, _bi_sv, _mu_sv = train_mf_sgd(
                _rows_k, n_factors=_MF_FACTORS, n_epochs=_MF_EPOCHS, seed=42)
            for _i in range(n_train):
                if _fold[_i] == _k:
                    oof_svd[_i] = mf_predict(
                        train_rows[_i][0], train_rows[_i][1],
                        _u2i_sv, _b2i_sv, _P, _Q, _bu_sv, _bi_sv, _mu_sv)
            print("  OOF SVD fold {}: {:.1f}s".format(_k, time.time() - t0))

        _u2i_final, _b2i_final, _P_f, _Q_f, _bu_f, _bi_f, _mu_f = train_mf_sgd(
            train_rows, n_factors=_MF_FACTORS, n_epochs=_MF_EPOCHS + 4, seed=42)
        print("Final SVD done: {:.1f}s".format(time.time() - t0))

    # -- Step 5: Build XGBoost training matrix --------------------------------
    # Feature layout (total 144 when _USE_SVD=True, 143 otherwise):
    #   112 biz  : 102 base + ck + tb + 6 photo + tip_avg_words + tip_avg_excl
    #   25  user : 23 base + tu + tip_avg_words
    #   6   pair : OOF u_avg, OOF i_avg, diff, OOF ucat, jaccard, ub_tip_flag
    #   [+3 if _USE_OOF_CF: cf_score, cf_k, is_cold]
    #   [+1 if _USE_SVD  : svd_score]
    # P_u / Q_i raw latent vectors are excluded: each OOF fold and the final
    # model use different random latent spaces → train/test incompatibility.

    X_train = []
    for _i, r in enumerate(train_rows):
        _ua  = oof_ua[_i]
        _ia  = oof_ia[_i]
        _cfk  = float(oof_cf_k[_i])
        _ubtip = 1.0 if (r[0], r[1]) in tip_ub_set else 0.0
        row   = (list(biz_map.get(r[1], b_default))
                 + list(usr_map.get(r[0], u_default))
                 + [_ua, _ia, _ua - _ia, oof_ucat[_i], jaccard_train[_i], _ubtip])
        if _USE_OOF_CF:
            row += [oof_cf[_i], _cfk, 1.0 if _cfk < 5 else 0.0]
        if _USE_SVD:
            row += [oof_svd[_i]]
        X_train.append([v for i, v in enumerate(row) if i not in _DROP_FEAT_IDX])

    # -- Step 6: Train XGBoost ------------------------------------------------
    # Params tuned by tune_xgb.py safe-mode (50 trials, lr≥0.03, n≤700) on 119-feat.
    # n=700 hits the Vocareum-safe cap; true optimal n may be slightly higher.
    reg = xgb.XGBRegressor(
        max_depth=7,
        learning_rate=0.03,
        n_estimators=700,
        subsample=0.75,
        colsample_bytree=0.6,
        min_child_weight=3,
        gamma=0.3,
        reg_alpha=1.0,
        reg_lambda=4.0,
        objective="reg:linear",   # Vocareum old XGBoost requires reg:linear
        nthread=4,
        verbosity=0,
        seed=42,
    )
    reg.fit(X_train, y_train)
    print("XGBoost done: {:.1f}s".format(time.time() - t0))

    # -- Step 7: Build X_test -------------------------------------------------

    X_test = []
    for uid, bid in test_pairs:
        _ua  = u_avg.get(uid, g_avg)
        _ia  = i_avg.get(bid, g_avg)
        _ubtip = 1.0 if (uid, bid) in tip_ub_set else 0.0
        # User-category preference from full training history
        _cats = biz_cats.get(bid, [])
        _vals = []
        if uid in u_cat_sum_all:
            for _ci in _cats:
                if u_cat_cnt_all[uid][_ci] > 0:
                    _vals.append(u_cat_sum_all[uid][_ci] / u_cat_cnt_all[uid][_ci])
        _ucat_t = sum(_vals) / len(_vals) if _vals else g_avg
        # Jaccard
        _uc = u_cats_set.get(uid, set())
        _bc = set(biz_cats.get(bid, []))
        _union = len(_uc | _bc)
        _jac = float(len(_uc & _bc)) / _union if _union > 0 else 0.0
        row  = (list(biz_map.get(bid, b_default))
                + list(usr_map.get(uid, u_default))
                + [_ua, _ia, _ua - _ia, _ucat_t, _jac, _ubtip])
        if _USE_OOF_CF:
            _cf_p, _cf_k = cf_predict(uid, bid, u2i, i2u, u_avg, i_avg, g_avg,
                                      sim_cache, top_n=30)
            row += [float(_cf_p), float(_cf_k), 1.0 if _cf_k < 5 else 0.0]
        if _USE_SVD and _u2i_final is not None:
            row += [mf_predict(uid, bid, _u2i_final, _b2i_final,
                               _P_f, _Q_f, _bu_f, _bi_f, _mu_f)]
        X_test.append([v for i, v in enumerate(row) if i not in _DROP_FEAT_IDX])

    xgb_preds = reg.predict(X_test)

    # -- Step 8: Write output -------------------------------------------------

    with open(output_file, "w") as out:
        out.write("user_id,business_id,prediction\n")
        for idx, ((uid, bid), xgb_p) in enumerate(zip(test_pairs, xgb_preds)):
            if _USE_OOF_CF:
                pred = float(xgb_p)
            else:
                cf_p, cf_k = cf_predict(uid, bid, u2i, i2u, u_avg, i_avg,
                                         g_avg, sim_cache, top_n=30)
                cw   = 0.10 if cf_k >= 15 else (0.05 if cf_k >= 5 else 0.0)
                pred = cw * cf_p + (1.0 - cw) * float(xgb_p)
            out.write("{},{},{}\n".format(uid, bid, _clamp(pred)))

    print("Duration: {:.1f}s".format(time.time() - t0))
    sc.stop()
