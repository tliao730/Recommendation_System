import json
from pyspark import SparkContext, SparkConf

# 初始化 SparkContext (競賽規範必須使用 RDD)
conf = SparkConf().setAppName("encode_business_rdd").setMaster("local[*]")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# ── 配置特徵清單 ──
TOP_STATES = ["AZ", "NV", "ON", "NC", "OH", "PA", "QC", "AB", "WI", "IL"]
TOP_CATEGORIES = [
    "Restaurants", "Shopping", "Food", "Beauty & Spas", "Home Services",
    "Health & Medical", "Local Services", "Automotive", "Nightlife", "Bars",
    "Event Planning & Services", "Active Life", "Fashion", "Coffee & Tea",
    "Sandwiches", "Hair Salons", "Fast Food", "American (Traditional)",
    "Pizza", "Home & Garden", "Hotels & Travel", "Arts & Entertainment",
    "Burgers", "Mexican", "Chinese", "Italian", "Japanese", "Sushi Bars",
    "Breakfast & Brunch", "Grocery",
]

# ── 輔助函式 ──
def parse_hours(hours_dict, day):
    if not hours_dict or day not in hours_dict:
        return -1.0
    try:
        hours_str = hours_dict[day]
        open_t, close_t = hours_str.split("-")
        oh, om = map(int, open_t.split(":"))
        ch, cm = map(int, close_t.split(":"))
        duration = (ch + cm / 60.0) - (oh + om / 60.0)
        return float(duration + 24 if duration < 0 else duration)
    except:
        return -1.0

def process_business_line(line):
    data = json.loads(line)
    attr = data.get("attributes") or {}
    
    # 建立特徵字典
    feat = {
        "business_id": data.get("business_id"),
        "stars": data.get("stars"),
        "review_count": data.get("review_count"),
        "is_open": data.get("is_open"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude")
    }

    # 1. Boolean 屬性 (1/0/-1)
    bool_attrs = [
        "BikeParking", "BusinessAcceptsCreditCards", "Caters", "CoatCheck",
        "DogsAllowed", "DriveThru", "GoodForDancing", "GoodForKids", "HappyHour",
        "HasTV", "Open24Hours", "OutdoorSeating", "RestaurantsDelivery",
        "RestaurantsGoodForGroups", "RestaurantsReservations", "RestaurantsTableService",
        "RestaurantsTakeOut", "WheelchairAccessible", "BYOB", "ByAppointmentOnly",
        "Corkage", "AcceptsInsurance"
    ]
    for b_attr in bool_attrs:
        val = attr.get(b_attr)
        feat[f"attr_{b_attr}"] = 1 if val == "True" else (0 if val == "False" else -1)

    # 2. PriceRange
    pr = attr.get("RestaurantsPriceRange2")
    feat["attr_PriceRange"] = int(pr) if pr in ["1", "2", "3", "4"] else -1

    # 3 & 4. Alcohol & WiFi (One-hot)
    alc = attr.get("Alcohol")
    feat["attr_Alcohol_none"] = 1 if alc == "none" else 0
    feat["attr_Alcohol_beer_and_wine"] = 1 if alc == "beer_and_wine" else 0
    feat["attr_Alcohol_full_bar"] = 1 if alc == "full_bar" else 0
    
    wifi = attr.get("WiFi")
    feat["attr_WiFi_no"] = 1 if wifi == "no" else 0
    feat["attr_WiFi_free"] = 1 if wifi == "free" else 0
    feat["attr_WiFi_paid"] = 1 if wifi == "paid" else 0

    # 5 & 6. Ordinal (Noise & Attire)
    noise_map = {"quiet": 0, "average": 1, "loud": 2, "very_loud": 3}
    feat["attr_NoiseLevel"] = noise_map.get(attr.get("NoiseLevel"), -1)
    
    attire_map = {"casual": 0, "dressy": 1, "formal": 2}
    feat["attr_Attire"] = attire_map.get(attr.get("RestaurantsAttire"), -1)

    # 7, 8, 9. Dict-like (Ambience, Parking, Meal)
    for prefix, keys, field in [
        ("ambience", ["romantic", "intimate", "classy", "hipster", "touristy", "trendy", "upscale", "casual"], "Ambience"),
        ("parking", ["garage", "street", "validated", "lot", "valet"], "BusinessParking"),
        ("meal", ["dessert", "latenight", "lunch", "dinner", "breakfast", "brunch"], "GoodForMeal")
    ]:
        f_str = attr.get(field, "")
        for k in keys:
            feat[f"{prefix}_{k}"] = 1 if f"'{k}': True" in str(f_str) else (0 if f"'{k}': False" in str(f_str) else -1)

    # 10. Hours
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        feat[f"hours_{day}_open"] = parse_hours(data.get("hours"), day)

    # 11. State (One-hot)
    b_state = data.get("state")
    for s in TOP_STATES:
        feat[f"state_{s}"] = 1 if b_state == s else 0

    # 12. Categories (Multi-hot)
    b_cats = data.get("categories") or ""
    for c in TOP_CATEGORIES:
        safe_name = c.replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")
        feat[f"cat_{safe_name}"] = 1 if c in b_cats else 0

    return feat

# ── 執行轉換 ──
raw_rdd = sc.textFile("./data/business.json")
feature_rdd = raw_rdd.map(process_business_line)

# ── 印出前 20 筆檢查 ──
# sample_20 = feature_rdd.take(20)

# print(f"Total processed: {feature_rdd.count()} rows")
# print("-" * 30)
# for i, row in enumerate(sample_20):
#     print(f"Record {i+1}: {row['business_id']} | Stars: {row['stars']} | Is_Open: {row['is_open']}")
#     # 這裡只印出部分特徵示意，你可以 row.keys() 查看完整列表
#     print(f"  Sample Features: State_AZ: {row.get('state_AZ')}, Cat_Restaurants: {row.get('cat_Restaurants')}")
#     print("-" * 10)

# sc.stop()
import pprint

# ── 執行轉換並取得第一筆資料 ──
# 確保全程使用 RDD 操作以符合競賽規範 [cite: 11, 124, 348]
first_record = feature_rdd.first()

print("\n" + "="*50)
print("UNIT TEST: COMPLETE FEATURE DATA FOR THE FIRST RECORD")
print("="*50)

# 使用 pprint 讓巢狀字典與多個特徵欄位易於閱讀
pprint.pprint(first_record, width=80, indent=2)

print("="*50)
print(f"Total Features Extracted: {len(first_record.keys())}")
print("="*50)