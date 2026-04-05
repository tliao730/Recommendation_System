import json
from datetime import datetime
from pyspark import SparkContext, SparkConf

# 初始化 SparkContext (必須使用 RDD) [cite: 11, 149]
conf = SparkConf().setAppName("encode_user_rdd").setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)
sc.setLogLevel("ERROR")

# 設定當前時間基準 (2026年4月)
CURRENT_DATE = datetime(2026, 4, 4)

def process_user_line(line):
    data = json.loads(line)
    
    # ── 1. 初始化字典與基本數值 ──
    # 使用 .get(key, 0) 確保即便 JSON 缺項也不會報錯 [cite: 14]
    res = {
        "user_id": data.get("user_id"),
        "review_count": data.get("review_count", 0),
        "average_stars": data.get("average_stars", 0.0),
        "fans": data.get("fans", 0),
        "useful": data.get("useful", 0),
        "funny": data.get("funny", 0),
        "cool": data.get("cool", 0)
    }

    # ── 2. yelping_since 處理 (修正日期格式問題) ──
    since_str = data.get("yelping_since")
    since_date = None
    if since_str:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                since_date = datetime.strptime(since_str, fmt)
                break # 成功解析就跳出循環
            except ValueError:
                continue
    
    if since_date:
        res["days_on_yelp"] = (CURRENT_DATE - since_date).days
        res["join_year"] = since_date.year
    else:
        res["days_on_yelp"] = 0
        res["join_year"] = 0

    # ── 3. 其他工程化特徵 ──
    # Friends
    friends = data.get("friends")
    res["friend_count"] = len(friends.split(",")) if friends and friends != "None" else 0
    
    # Elite
    elite = data.get("elite")
    if elite and elite != "None":
        res["is_elite"] = 1
        res["elite_year_count"] = len(elite.split(","))
    else:
        res["is_elite"] = 0
        res["elite_year_count"] = 0

    # Compliments (11項)
    compliment_keys = [
        "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute",
        "compliment_list", "compliment_note", "compliment_plain", "compliment_cool",
        "compliment_funny", "compliment_writer", "compliment_photos"
    ]
    total_c = 0
    for key in compliment_keys:
        val = data.get(key, 0)
        res[key] = val
        total_c += val
    res["total_compliments"] = total_c

    # ── 4. Engagement score (確保在這裡計算並賦值) ──
    res["engagement_score"] = res["useful"] + res["funny"] + res["cool"]

    # 重要：確保 return 在函式的最後一行，且不在任何 if/except 區塊內
    return res

# ── 執行轉換 ──
raw_user_rdd = sc.textFile("data/user.json")
user_feature_rdd = raw_user_rdd.map(process_user_line)

# ── 印出前 5 筆檢查 ──
sample_data = user_feature_rdd.take(5)

print("\n" + "="*60)
print("UNIT TEST: USER FEATURES (FIRST 5 RECORDS)")
print("="*60)

for i, user in enumerate(sample_data):
    print(f"User {i+1}: {user['user_id']}")
    print(f"  Stars: {user['average_stars']} | Reviews: {user['review_count']}")
    print(f"  Engagement: {user['engagement_score']} | Total Compliments: {user['total_compliments']}")
    print(f"  Elite Status: {user['is_elite']} (Years: {user['elite_year_count']})")
    print(f"  Yelp Tenure: {user['days_on_yelp']} days (Joined: {user['join_year']})")
    print("-" * 40)

# 如果要檢查特定數值統計，可以這樣寫：
avg_stars_stats = user_feature_rdd.map(lambda x: x['average_stars']).stats()
print(f"\nAverage Stars Statistics:\n{avg_stars_stats}")

sc.stop()