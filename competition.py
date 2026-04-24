import os
import sys
import time
import json
import math
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
    train_path = os.path.join(folder_path, "yelp_train.csv")

    # -- Step 1: Load & broadcast auxiliary data via RDD ----------------------
    # Each textFile, map, collectAsMap is a full parallel Spark job.

    # 1a. Business: 102 raw features per business
    biz_raw = sc.textFile(biz_path).map(extract_business).collectAsMap()

    # 1b. Checkin: sum of all checkin counts per business (old Yelp format)
    ck_map = (sc.textFile(ck_path)
                .map(extract_checkin)
                .reduceByKey(lambda a, b: a + b)
                .collectAsMap())

    # 1c. Tip: one pass: split into per-business and per-user counts
    tip_all = (sc.textFile(tip_path)
                 .flatMap(extract_tip)
                 .reduceByKey(lambda a, b: a + b)
                 .collectAsMap())
    tb_map = {k[2:]: v for k, v in tip_all.items() if k[:2] == "b_"}
    tu_map = {k[2:]: v for k, v in tip_all.items() if k[:2] == "u_"}

    # 1d. User: 23 raw features per user
    usr_raw = sc.textFile(usr_path).map(extract_user).collectAsMap()

    # Merge auxiliary counts into feature lists
    # Business: 102, checkin_cnt, tip_biz_cnt = 104 features
    biz_map = {bid: f + [ck_map.get(bid, 0.0), tb_map.get(bid, 0.0)]
               for bid, f in biz_raw.items()}

    # User: 23 and tip_usr_cnt = 24 features
    usr_map = {uid: f + [tu_map.get(uid, 0.0)]
               for uid, f in usr_raw.items()}

    # Cold-start defaults: per-column mean (ignoring -1 sentinel for missing attrs)
    biz_vals  = list(biz_map.values())
    n_bfeat   = len(biz_vals[0]) if biz_vals else 104
    b_default = _col_defaults(biz_vals, n_bfeat)

    usr_vals  = list(usr_map.values())
    n_ufeat   = len(usr_vals[0]) if usr_vals else 24
    u_default = _col_defaults(usr_vals, n_ufeat)

    # -- Step 2: Load & persist train RDD -------------------------------------
    train_raw = sc.textFile(train_path)
    hdr       = train_raw.first()
    train_rdd = (train_raw
                 .filter(lambda x: x != hdr)
                 .map(lambda s: s.split(","))
                 .persist())

    # -- Step 3: Build CF structures (2 RDD passes on persisted RDD) ----------
    # Only string/float data flows through Spark here (small), no feature vecs.
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

    # -- Step 4: Build XGBoost training data (Python-side, no JVM) -----------
    # Collect only lightweight string triples (~22 MB), then do dict lookups
    # in Python. Avoids broadcasting large feature maps through JVM heap.
    train_rows = train_rdd.collect()
    X_train = [list(biz_map.get(r[1], b_default)) + list(usr_map.get(r[0], u_default))
               for r in train_rows]
    y_train = [float(r[2]) for r in train_rows]

    # -- Step 5: Train XGBoost -------------------------------------------------
    reg = xgb.XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:linear",
        nthread=4,
        verbosity=0,
        seed=42,
    )
    reg.fit(X_train, y_train)

    # -- Step 6: Load test pairs via RDD ---------------------------------------
    test_raw   = sc.textFile(test_file)
    test_hdr   = test_raw.first()
    test_pairs = (test_raw
                  .filter(lambda x: x != test_hdr)
                  .map(lambda s: s.split(","))
                  .map(lambda p: (p[0], p[1]))
                  .collect())

    # -- Step 7: Batch XGBoost prediction -------------------------------------
    X_test    = [list(biz_map.get(bid, b_default)) + list(usr_map.get(uid, u_default))
                 for uid, bid in test_pairs]
    xgb_preds = reg.predict(X_test)

    # -- Step 8: Hybrid blend & write output -----------------------------------
    # CF weight scales with neighbor count:
    
    with open(output_file, "w") as out:
        out.write("user_id,business_id,prediction\n")
        for (uid, bid), xgb_p in zip(test_pairs, xgb_preds):
            cp, ck = cf_predict(uid, bid, u2i, i2u, u_avg, i_avg, g_avg, sim_cache, top_n=30)

            if ck >= 10:
                cw = 0.35
            elif ck >= 3:
                cw = 0.25
            elif ck >= 1:
                cw = 0.10
            else:
                cw = 0.0

            pred = _clamp(cw * cp + (1.0 - cw) * float(xgb_p))
            out.write("{},{},{}\n".format(uid, bid, pred))

    print("Duration: {:.1f}s".format(time.time() - t0))
    sc.stop()
