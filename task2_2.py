import json
import os
import sys
import time
import xgboost as xgb

from pyspark import SparkContext

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def extract_business(line):
    """Parse business.json and extract features"""
    data = json.loads(line)
    bid = data.get('business_id', '')
    stars = float(data.get('stars', 0.0))
    rev_count = float(data.get('review_count', 0.0))
    is_open = float(data.get('is_open', 1.0))
    
    # Defaults for attributes
    price = 2.0 
    good_for_kids = 0.0
    delivery = 0.0
    
    attrs = data.get('attributes')
    if attrs:
        # 1. Price Range
        if 'RestaurantsPriceRange2' in attrs and attrs['RestaurantsPriceRange2'] not in (None, 'None'):
            try:
                price = float(attrs['RestaurantsPriceRange2'])
            except ValueError:
                pass
        
        # 2. GoodForKids
        if 'GoodForKids' in attrs and attrs['GoodForKids'] == 'True':
            good_for_kids = 1.0
            
        # 3. Delivery
        if 'RestaurantsDelivery' in attrs and attrs['RestaurantsDelivery'] == 'True':
            delivery = 1.0
                
    return (bid, (stars, rev_count, price, is_open, good_for_kids, delivery))

def extract_user(line):
    """Parse user.json and extract features"""
    data = json.loads(line)
    uid = data.get('user_id', '')
    rev_count = float(data.get('review_count', 0.0))
    useful = float(data.get('useful', 0.0))
    fans = float(data.get('fans', 0.0))
    avg_stars = float(data.get('average_stars', 0.0))
    
    # 1. Elite Count (Number of years being elite)
    elite_str = data.get('elite', '')
    elite_count = 0.0
    if elite_str and elite_str != 'None':
        elite_count = float(len(elite_str.split(',')))
        
    # 2. Account Age (Approximation based on 'yelping_since' year)
    yelping_since = data.get('yelping_since', '2015')
    try:
        # e.g., "2015-09-28" -> 2015. Account age relative to 2025.
        account_age = 2025.0 - float(yelping_since[:4])
    except:
        account_age = 10.0 # Default fallback
        
    # 3. Total Compliments (Sum of all compliment types)
    compliments = sum([
        float(data.get('compliment_hot', 0)),
        float(data.get('compliment_more', 0)),
        float(data.get('compliment_profile', 0)),
        float(data.get('compliment_cute', 0)),
        float(data.get('compliment_list', 0)),
        float(data.get('compliment_note', 0)),
        float(data.get('compliment_plain', 0)),
        float(data.get('compliment_cool', 0)),
        float(data.get('compliment_funny', 0)),
        float(data.get('compliment_writer', 0)),
        float(data.get('compliment_photos', 0))
    ])
    
    return (uid, (rev_count, useful, fans, avg_stars, elite_count, account_age, compliments))

if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize SparkContext
    sc = SparkContext('local[*]', 'task2_2')
    sc.setLogLevel("WARN")

    # Handle input arguments
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    train_file_path = os.path.join(folder_path, "yelp_train.csv")
    business_file_path = os.path.join(folder_path, "business.json")
    user_file_path = os.path.join(folder_path, "user.json")

    # 1. Process Business Data
    business_rdd = sc.textFile(business_file_path).map(extract_business)
    business_map = business_rdd.collectAsMap()
    
    # Calculate Business Averages for cold-start filling
    b_vals = list(business_map.values())
    b_len = len(b_vals) if b_vals else 1
    avg_b_stars = sum(v[0] for v in b_vals) / b_len
    avg_b_rev = sum(v[1] for v in b_vals) / b_len
    avg_b_price = sum(v[2] for v in b_vals) / b_len
    avg_b_open = sum(v[3] for v in b_vals) / b_len
    avg_b_kids = sum(v[4] for v in b_vals) / b_len
    avg_b_deliv = sum(v[5] for v in b_vals) / b_len
    b_default = (avg_b_stars, avg_b_rev, avg_b_price, avg_b_open, avg_b_kids, avg_b_deliv)

    # 2. Process User Data
    user_rdd = sc.textFile(user_file_path).map(extract_user)
    user_map = user_rdd.collectAsMap()
    
    # Calculate User Averages for cold-start filling
    u_vals = list(user_map.values())
    u_len = len(u_vals) if u_vals else 1
    avg_u_rev = sum(v[0] for v in u_vals) / u_len
    avg_u_useful = sum(v[1] for v in u_vals) / u_len
    avg_u_fans = sum(v[2] for v in u_vals) / u_len
    avg_u_stars = sum(v[3] for v in u_vals) / u_len
    avg_u_elite = sum(v[4] for v in u_vals) / u_len
    avg_u_age = sum(v[5] for v in u_vals) / u_len
    avg_u_comp = sum(v[6] for v in u_vals) / u_len
    u_default = (avg_u_rev, avg_u_useful, avg_u_fans, avg_u_stars, avg_u_elite, avg_u_age, avg_u_comp)

    # Broadcast dictionaries to worker nodes
    b_map_br = sc.broadcast(business_map)
    u_map_br = sc.broadcast(user_map)

    # Helper function to construct feature vectors
    def build_features(row, is_train=True):
        uid = row[0]
        bid = row[1]
        
        # Get features from broadcast variables, fallback to defaults if missing
        b_feat = b_map_br.value.get(bid, b_default)
        u_feat = u_map_br.value.get(uid, u_default)
        
        # Combine into a single list: 6 business features + 7 user features = 13 features
        features = list(b_feat) + list(u_feat)
        
        if is_train:
            label = float(row[2])
            return features, label
        else:
            return uid, bid, features

    # 3. Process Train Data
    train_rdd_raw = sc.textFile(train_file_path)
    train_header = train_rdd_raw.first()
    train_data = (train_rdd_raw.filter(lambda x: x != train_header)
                               .map(lambda x: x.split(','))
                               .map(lambda row: build_features(row, is_train=True))
                               .collect())
    
    X_train = [x[0] for x in train_data]
    y_train = [x[1] for x in train_data]

    # 4. Process Test Data
    test_rdd_raw = sc.textFile(test_file_name)
    test_header = test_rdd_raw.first()
    test_data = (test_rdd_raw.filter(lambda x: x != test_header)
                             .map(lambda x: x.split(','))
                             .map(lambda row: build_features(row, is_train=False))
                             .collect())

    test_uids = [x[0] for x in test_data]
    test_bids = [x[1] for x in test_data]
    X_test = [x[2] for x in test_data]

    # 5. Train XGBoost Model with Optimized Hyperparameters (XGB 0.72 compatible)
    xgboost_model = xgb.XGBRegressor(
        max_depth=6,              # 增加一點深度來捕捉複雜特徵，但不過深以免過擬合
        learning_rate=0.05,       # 降低學習率讓收斂更穩定
        n_estimators=300,         # 配合低學習率，提高樹的數量
        subsample=0.8,            # 每次建樹只用 80% 資料 (防過擬合)
        colsample_bytree=0.8,     # 每次建樹只用 80% 特徵 (防過擬合)
        reg_alpha=0.1,            # L1 正則化 (Lasso)
        reg_lambda=2.0,           # L2 正則化 (Ridge)，加強數值穩定性
        silent=True,              # 關閉除錯日誌
        objective='reg:linear',   # 0.72 版本必須指定的 Loss Function
        seed=42
    )
    xgboost_model.fit(X_train, y_train)

    # 6. Predict
    predictions = xgboost_model.predict(X_test)

    # 7. Write to CSV
    with open(output_file_name, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for i in range(len(predictions)):
            f.write(f"{test_uids[i]},{test_bids[i]},{float(predictions[i])}\n")

    sc.stop()
    # print(f"Duration: {time.time() - start_time}")