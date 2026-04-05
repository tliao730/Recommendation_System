import os
import sys
import time
import json
import math
from pyspark.context import SparkContext


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def clamp_rating(x):
    if x < 1.0:
        return 1.0
    if x > 5.0:
        return 5.0
    return x


# -------------------- Feature extraction (match your task2_2 style) --------------------
def extract_business(line):
    data = json.loads(line)
    bid = data.get("business_id", "")
    stars = safe_float(data.get("stars", 0.0), 0.0)
    rev_count = safe_float(data.get("review_count", 0.0), 0.0)
    is_open = safe_float(data.get("is_open", 1.0), 1.0)

    price = 2.0
    good_for_kids = 0.0
    delivery = 0.0
    attrs = data.get("attributes")
    if attrs:
        v = attrs.get("RestaurantsPriceRange2")
        if v not in (None, "None"):
            try:
                price = float(v)
            except Exception:
                pass
        if attrs.get("GoodForKids") == "True":
            good_for_kids = 1.0
        if attrs.get("RestaurantsDelivery") == "True":
            delivery = 1.0

    return bid, (stars, rev_count, price, is_open, good_for_kids, delivery)


def extract_user(line):
    data = json.loads(line)
    uid = data.get("user_id", "")
    rev_count = safe_float(data.get("review_count", 0.0), 0.0)
    useful = safe_float(data.get("useful", 0.0), 0.0)
    fans = safe_float(data.get("fans", 0.0), 0.0)
    avg_stars = safe_float(data.get("average_stars", 0.0), 0.0)

    elite_str = data.get("elite", "")
    elite_count = 0.0
    if elite_str and elite_str != "None":
        elite_count = float(len(elite_str.split(",")))

    yelping_since = data.get("yelping_since", "2015")
    try:
        account_age = 2026.0 - float(yelping_since[:4])
    except Exception:
        account_age = 10.0

    compliments = 0.0
    for k in (
        "compliment_hot",
        "compliment_more",
        "compliment_profile",
        "compliment_cute",
        "compliment_list",
        "compliment_note",
        "compliment_plain",
        "compliment_cool",
        "compliment_funny",
        "compliment_writer",
        "compliment_photos",
    ):
        compliments += safe_float(data.get(k, 0.0), 0.0)

    return uid, (rev_count, useful, fans, avg_stars, elite_count, account_age, compliments)


def build_dense_features(uid, bid, b_map, u_map, b_default, u_default):
    b_feat = b_map.get(bid, b_default)
    u_feat = u_map.get(uid, u_default)
    # 6 business + 7 user = 13 features
    return list(b_feat) + list(u_feat)


# -------------------- Item-based CF (from your task2_1) --------------------
def generate_rating(input_rdd, main_idx, sub_idx, val_idx):
    return (
        input_rdd.map(lambda x: (x[main_idx], (x[sub_idx], float(x[val_idx]))))
        .groupByKey()
        .mapValues(dict)
        .collectAsMap()
    )


def generate_avg_rating(input_rdd, key_idx, val_idx):
    return (
        input_rdd.map(lambda x: (x[key_idx], float(x[val_idx])))
        .groupByKey()
        .mapValues(lambda rs: sum(rs) / len(rs))
        .collectAsMap()
    )


def pearson_sim(i, j, item_to_user, item_avg, min_common=2, sim_cache=None):
    if sim_cache is not None:
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key in sim_cache:
            return sim_cache[key]
    else:
        key = None

    if i not in item_to_user or j not in item_to_user:
        sim = 0.0
    else:
        ui = item_to_user[i]
        uj = item_to_user[j]
        common = set(ui.keys()) & set(uj.keys())
        if len(common) < min_common:
            sim = 0.0
        else:
            avg_i = item_avg.get(i, 0.0)
            avg_j = item_avg.get(j, 0.0)
            num = 0.0
            di2 = 0.0
            dj2 = 0.0
            for u in common:
                ri = ui[u]
                rj = uj[u]
                di = ri - avg_i
                dj = rj - avg_j
                num += di * dj
                di2 += di * di
                dj2 += dj * dj
            denom = math.sqrt(di2) * math.sqrt(dj2)
            sim = 0.0 if denom == 0.0 else (num / denom)
            # shrinkage
            shrink = len(common) / (len(common) + 2.0)
            sim *= shrink

    if sim_cache is not None and key is not None:
        sim_cache[key] = sim
    return sim


def cf_predict_with_conf(uid, bid, user_to_item, item_to_user, user_avg, item_avg, global_avg, sim_cache, top_n=60):
    # cold start
    if uid not in user_to_item and bid not in item_to_user:
        return global_avg, 0.0, 0
    if uid not in user_to_item:
        return item_avg.get(bid, global_avg), 0.0, 0
    if bid not in item_to_user:
        return user_avg.get(uid, global_avg), 0.0, 0

    rated = user_to_item[uid]
    if bid in rated:
        return float(rated[bid]), 999.0, 1  # very confident

    sims = []
    for nb in rated.keys():
        if nb == bid:
            continue
        s = pearson_sim(bid, nb, item_to_user, item_avg, min_common=2, sim_cache=sim_cache)
        if s >= 0:
            sims.append((nb, s))

    if not sims:
        base = item_avg.get(bid, user_avg.get(uid, global_avg))
        return clamp_rating(base), 0.0, 0

    sims.sort(key=lambda x: abs(x[1]), reverse=True)
    top = sims[:top_n]

    numerator = 0.0
    denom = 0.0
    b_ui = 0.15 * user_avg.get(uid, global_avg) + 0.85 * item_avg.get(bid, global_avg)
    for nb, s in top:
        r_un = float(rated[nb])
        b_un = 0.15 * user_avg.get(uid, global_avg) + 0.85 * item_avg.get(nb, global_avg)
        numerator += s * (r_un - b_un)
        denom += abs(s)

    if denom == 0.0:
        pred = item_avg.get(bid, user_avg.get(uid, global_avg))
    else:
        cf_pred = b_ui + numerator / denom
        w = denom / (denom + 3.0)
        pred = w * cf_pred + (1.0 - w) * b_ui

    return clamp_rating(pred), denom, len(top)


def deterministic_holdout(uid, bid, frac=0.2):
    # deterministic split without random lib
    return (abs(hash(uid + "||" + bid)) % 1000) < int(frac * 1000)


if __name__ == "__main__":
    start_time = time.time()

    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    sc = SparkContext("local[*]", "task2_3")
    sc.setLogLevel("ERROR")

    # ---------- Load aux data ----------
    business_path = os.path.join(folder_path, "business.json")
    user_path = os.path.join(folder_path, "user.json")
    train_path = os.path.join(folder_path, "yelp_train.csv")

    business_map = sc.textFile(business_path).map(extract_business).collectAsMap()
    user_map = sc.textFile(user_path).map(extract_user).collectAsMap()

    b_vals = list(business_map.values())
    bl = len(b_vals) if b_vals else 1
    b_default = (
        sum(v[0] for v in b_vals) / bl,
        sum(v[1] for v in b_vals) / bl,
        sum(v[2] for v in b_vals) / bl,
        sum(v[3] for v in b_vals) / bl,
        sum(v[4] for v in b_vals) / bl,
        sum(v[5] for v in b_vals) / bl,
    )

    u_vals = list(user_map.values())
    ul = len(u_vals) if u_vals else 1
    u_default = (
        sum(v[0] for v in u_vals) / ul,
        sum(v[1] for v in u_vals) / ul,
        sum(v[2] for v in u_vals) / ul,
        sum(v[3] for v in u_vals) / ul,
        sum(v[4] for v in u_vals) / ul,
        sum(v[5] for v in u_vals) / ul,
        sum(v[6] for v in u_vals) / ul,
    )

    b_map_br = sc.broadcast(business_map)
    u_map_br = sc.broadcast(user_map)

    # ---------- Load train (RDD) ----------
    train_lines = sc.textFile(train_path)
    train_header = train_lines.first()
    train_raw = train_lines.filter(lambda x: x != train_header).map(lambda s: s.split(",")).persist()

    # ---------- Build CF structures from full train ----------
    user_to_item = generate_rating(train_raw, 0, 1, 2)
    item_to_user = generate_rating(train_raw, 1, 0, 2)
    user_avg = generate_avg_rating(train_raw, 0, 2)
    item_avg = generate_avg_rating(train_raw, 1, 2)
    global_avg = train_raw.map(lambda x: float(x[2])).mean()
    sim_cache = {}

    # ---------- Train model-based regressor (XGBRegressor) ----------
    import xgboost as xgb

    def to_model_xy(row):
        uid, bid, stars = row[0], row[1], float(row[2])
        feat = build_dense_features(uid, bid, b_map_br.value, u_map_br.value, b_default, u_default)
        return feat, stars

    train_xy = train_raw.map(to_model_xy).collect()
    X_train = [x for x, _ in train_xy]
    y_train = [y for _, y in train_xy]

    reg = xgb.XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:linear",
        seed=42,
        nthread=4,
        verbosity=0,
    )
    reg.fit(X_train, y_train)

    # ---------- Train meta-classifier (Method 2) using holdout from yelp_train ----------
    holdout = [t for t in train_xy if deterministic_holdout(str(id(t)), str(id(t)), frac=0.0)]  # placeholder
    # Build holdout deterministically from raw strings to avoid driver randomness
    train_triplets = train_raw.map(lambda r: (r[0], r[1], float(r[2]))).collect()
    holdout_triplets = [t for t in train_triplets if deterministic_holdout(t[0], t[1], frac=0.2)]
    if len(holdout_triplets) < 1000:
        # fallback: smaller split if hash distribution odd
        holdout_triplets = [t for t in train_triplets if deterministic_holdout(t[0], t[1], frac=0.1)]

    X_meta = []
    y_meta = []
    for uid, bid, true_r in holdout_triplets:
        # base features (13)
        base_feat = build_dense_features(uid, bid, business_map, user_map, b_default, u_default)
        # CF prediction + confidence
        cf_p, cf_denom, cf_k = cf_predict_with_conf(
            uid, bid, user_to_item, item_to_user, user_avg, item_avg, global_avg, sim_cache, top_n=60
        )
        # model prediction
        m_p = clamp_rating(float(reg.predict([base_feat])[0]))
        # label: 1 if CF closer else 0
        if abs(true_r - cf_p) <= abs(true_r - m_p):
            y = 1
        else:
            y = 0
        # meta features: base + cf confidence + simple diffs
        X_meta.append(base_feat + [cf_denom, float(cf_k), cf_p, m_p, abs(cf_p - m_p)])
        y_meta.append(y)

    clf = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=150,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        seed=42,
        nthread=4,
        verbosity=0,
    )
    # If holdout too small (edge case), fall back to always model-based
    use_classifier = len(set(y_meta)) > 1 and len(y_meta) >= 500
    if use_classifier:
        clf.fit(X_meta, y_meta)

    # ---------- Predict on test ----------
    test_lines = sc.textFile(test_file)
    test_header = test_lines.first()
    test_pairs = test_lines.filter(lambda x: x != test_header).map(lambda s: s.split(",")).map(lambda p: (p[0], p[1])).collect()

    with open(output_file, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for uid, bid in test_pairs:
            base_feat = build_dense_features(uid, bid, business_map, user_map, b_default, u_default)
            cf_p, cf_denom, cf_k = cf_predict_with_conf(
                uid, bid, user_to_item, item_to_user, user_avg, item_avg, global_avg, sim_cache, top_n=60
            )
            m_p = clamp_rating(float(reg.predict([base_feat])[0]))

            if use_classifier:
                meta_feat = base_feat + [cf_denom, float(cf_k), cf_p, m_p, abs(cf_p - m_p)]
                prob_cf = float(clf.predict_proba([meta_feat])[0][1])
                # hard decision (method 2), but softly blended for stability
                pred = prob_cf * cf_p + (1.0 - prob_cf) * m_p
            else:
                pred = m_p

            f.write("{},{},{}\n".format(uid, bid, clamp_rating(float(pred))))

    print("Duration: {:.2f} seconds".format(time.time() - start_time))
    sc.stop()

