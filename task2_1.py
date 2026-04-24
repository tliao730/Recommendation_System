import os 
import time
import sys
import math
from pyspark.context import SparkContext

# Environment and Spark Initialization
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext("local[*]", "task2_1")
sc.setLogLevel("ERROR")

def generate_rating(input_rdd, MainKey_index, SubKey_index, value_index):
    output_dict = input_rdd.map(lambda x: (x[MainKey_index], (x[SubKey_index], float(x[value_index])))) \
                                .groupByKey() \
                                .mapValues(dict) \
                                .collectAsMap()
    return output_dict

def generate_avg_rating(input_rdd, key_index, value_index):
    output_dict = input_rdd.map(lambda x: (x[key_index], float(x[value_index]))) \
                        .groupByKey() \
                        .mapValues(lambda x: sum(x)/len(x)) \
                        .collectAsMap()
    return output_dict

def generate_pearson_correlation(item_i, item_j, item_to_user_dict, item_avg_dict, min_common_users = 2, sim_cache = None):

    # 1. Check if the similarity is already calculated and cached.
    if sim_cache is not None:
        a, b = (item_i, item_j) if item_i < item_j else (item_j, item_i)
        key = (a, b)
        if key in sim_cache:
            return sim_cache[key]
    else:
        key = None

    # 2. Is the item in the dictionary?
    if item_i not in item_to_user_dict or item_j not in item_to_user_dict:
        sim = 0.0
    else:
        user_i = item_to_user_dict[item_i] # dict: user -> rating
        user_j = item_to_user_dict[item_j] # dict: user -> rating

        # 3. common users set
        common_users = set(user_i.keys()) & set(user_j.keys())
        if len(common_users) < min_common_users:
            sim = 0.0
        else:
            avg_i = item_avg_dict[item_i]
            avg_j = item_avg_dict[item_j]
        
            # 4. Pearson : numerator / (sqrt(denom_i) * sqrt(demon_j))
            num = 0.0
            demon_i = 0.0
            demon_j = 0.0

            for u in common_users:
                r_i = float(user_i[u])
                r_j = float(user_j[u])

                di = r_i - avg_i
                dj = r_j - avg_j

                num += di * dj
                demon_i += di * di
                demon_j += dj * dj
            
            demon = math.sqrt(demon_i) * math.sqrt(demon_j)
            sim = 0.0 if demon == 0.0 else num / demon

            # add shrinkage here
            shrink = len(common_users) / (len(common_users) + 2.0)
            sim *= shrink
    
    # 5.
    if sim_cache is not None and key is not None:
        sim_cache[key] = sim
    
    return sim

def ItemBased_cf(user_id, business_id, user_to_item_dict, item_to_user_dict, user_avg_dict, item_avg_dict, global_avg, sim_cache, top_n = 50):
    # 1) cold-start
    if user_id not in user_to_item_dict and business_id not in item_to_user_dict: # both not in the dictionary
        return global_avg
    
    if user_id not in user_to_item_dict:
        return item_avg_dict[business_id]
    
    if business_id not in item_to_user_dict:
        return user_avg_dict[user_id]
    
    # 2) gather candidate neighbors from user history
    rated_items = user_to_item_dict[user_id]

    # if user already rated this business in train, return it directly
    if business_id in rated_items:
        return rated_items[business_id]
    
    neighbor_items = rated_items.keys()

    # 3) compute similarities with pearson
    similarity_list = []
    for neighbor_item in neighbor_items:
        if neighbor_item == business_id:
            continue
        
        sim = generate_pearson_correlation(
            business_id, neighbor_item, 
            item_to_user_dict, item_avg_dict,
            min_common_users = 2,
            sim_cache = sim_cache
        )

        # only keep positive similarities. (temporary)
        if sim >= 0:
            similarity_list.append((neighbor_item, sim))
    
    # 4) filter + sort + top_n
    similarity_sorted = sorted(similarity_list, key = lambda x: abs(x[1]), reverse = True)
    top_neighbors  = similarity_sorted[:top_n]

    # 5) weighted average + fallback + clip
    if len(top_neighbors) == 0:
        # no usable neighbors -> fallback
        base = item_avg_dict.get(business_id, user_avg_dict.get(user_id, global_avg))
        return max(1.0, min(5.0, base))

    numerator = 0.0
    denominator = 0.0
    b_ui = 0.15 * user_avg_dict.get(user_id, global_avg) + 0.85 * item_avg_dict.get(business_id, global_avg)
    for neighbor_item, sim in top_neighbors:
        r_u_n = float(rated_items[neighbor_item])
        b_u_n = 0.15 * user_avg_dict.get(user_id, global_avg) + 0.85 * item_avg_dict.get(neighbor_item, global_avg)
        numerator += sim * (r_u_n - b_u_n)
        denominator += abs(sim)
    if denominator == 0.0:
        pred = item_avg_dict.get(business_id, user_avg_dict.get(user_id, global_avg))
    else:
        cf_pred = b_ui + numerator / denominator
        w = denominator / (denominator + 3.0)   # 3.0 可試 2.0~5.0
        pred = w * cf_pred + (1.0 - w) * b_ui

    
    pred = max(1.0, min(5.0, pred))
    return pred
    
# =============================== Main Function ===============================
if __name__ == "__main__":
    start_time = time.time()
    input_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    # 1. Load data and preprocess
    lines = sc.textFile(input_file)
    header = lines.first()
    raw_rdd = lines.filter(lambda x: x != header).map(lambda x: x.split(","))

    # 2. generate rating matrix for user-item and item-user(use dictionary to pretend the matrix).
    # dict fit the cross comparison between "user and item", and "item and user" than rdd.
    user_to_item_dict = generate_rating(raw_rdd, 0, 1, 2)
    item_to_user_dict = generate_rating(raw_rdd, 1, 0, 2)

    # 3. generate average rating for each user and item.
    user_avg_dict = generate_avg_rating(raw_rdd, 0, 2)
    item_avg_dict = generate_avg_rating(raw_rdd, 1, 2)
    global_avg = raw_rdd.map(lambda x: float(x[2])).mean()

    # initialize similarity cache
    sim_cache = {}

    # 4. load test data and predict ratings
    test_lines = sc.textFile(test_file)
    test_header = test_lines.first()
    test_rdd = test_lines.filter(lambda x: x!= test_header).map(lambda x: x.split(","))

    # 5. predict ratings for test data
    predictions = test_rdd.map(lambda x: (
        x[0],
        x[1],
        ItemBased_cf(   # <- ItemBased_cf function
            x[0], x[1],
            user_to_item_dict, item_to_user_dict,
            user_avg_dict, item_avg_dict, global_avg, sim_cache, top_n = 60
        ) # <- ItemBased_cf function
    )).collect() # <- collect() function
    

    # 6. write predictions to output file
    with open(output_file, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for u, b, p in predictions:
            f.write("{},{},{}\n".format(u, b, p))

    print(f"Duration: {time.time() - start_time:.2f} seconds")