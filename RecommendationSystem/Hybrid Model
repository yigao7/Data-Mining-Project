# Combine XGBoost and collaborative filtering

from pyspark import SparkContext
import json
import pandas as pd
import xgboost as xgb
from math import sqrt
import time
import sys

def cal_pearson_sim(pair, common_users, r):
    i, j = pair[0], pair[1]
    ri = [r[(u, i)] for u in common_users]
    avg_ri = sum(ri)/len(ri)
    rj = [r[(u, j)] for u in common_users]
    avg_rj = sum(rj) / len(rj)
    numerator = sum([(r[(u, i)]-avg_ri) * (r[(u, j)]-avg_rj) for u in common_users])
    denominator = sqrt(sum([(r[(u, i)]-avg_ri)**2 for u in common_users])) * sqrt(sum([(r[(u, j)]-avg_rj)**2 for u in common_users]))
    if denominator == 0:
        weight = 0
    else:
        weight = numerator / denominator
    return (pair, weight)

def make_final_pred(u, i, MB_pred_dict, w, user_bus_dict, r):
    record = MB_pred_dict[(u, i)]
    pred_MB = record[0]
    if u not in user_bus_dict:   # cold start
        pred = pred_MB
    else:
        N = set()
        for n in user_bus_dict[u]:
            if tuple(sorted((i,n))) in w and w[tuple(sorted((i,n)))] > 0:
                N.add(n)
        if N == set():      # no weight calculated, or no similar businesses, to form neighborhood
            pred = pred_MB
        else:
            numerator = sum([r[(u,n)]*w[tuple(sorted((i,n)))] for n in N])
            denominator = sum([abs(w[tuple(sorted((i,n)))]) for n in N])
            pred_CF = numerator / denominator
            ### decide weights between two CF and MB
            CF_weight = len(N)*60
            MB_weight = record[1] + record[2]
            alpha = CF_weight / (CF_weight+MB_weight)
            pred = alpha * pred_CF + (1-alpha) * pred_MB
    return (u, i, pred_MB)

def gen_bus_features(x):
    categories = x[1][2]
    if type(categories) == str and 'Restaurants' in categories:
        restaurant = 1
    else:
        restaurant = 0
    return (x[0], (x[1][0], x[1][1], restaurant,x[1][3]))

if __name__ == '__main__':
    begin = time.time()

    sc = SparkContext('local[*]', 'task2.3')
    sc.setLogLevel("ERROR")

    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    # Read in all data
    business_filepath = folder_path + "/business.json"
    user_filepath = folder_path + "/user.json"
    photo_filepath = folder_path + "/photo.json"
    tip_filepath = folder_path + "/tip.json"
    checkin_filepath = folder_path + "/checkin.json"
    train_file_name = folder_path + "/yelp_train.csv"

    ### Business
    business_rdd = sc.textFile(business_filepath)
    business_info = business_rdd.map(lambda row: json.loads(row)) \
                                .map(lambda x: (x['business_id'], (x['stars'], x['review_count'], x['categories'], x['state']))) \
                                .map(lambda x: gen_bus_features(x))

    ### User
    user_rdd = sc.textFile(user_filepath)
    user_info = user_rdd.map(lambda row: json.loads(row)) \
                        .map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'], x['useful'])))

    ### Photo
    photo_rdd = sc.textFile(photo_filepath)
    photo_cnt = photo_rdd.map(lambda row: json.loads(row)) \
                         .map(lambda x: (x['business_id'], 1)) \
                         .groupByKey() \
                         .mapValues(lambda x: len(x))

    ### Tip
    tip_rdd = sc.textFile(tip_filepath)
    bus_tip_cnt = tip_rdd.map(lambda row: json.loads(row)) \
                         .map(lambda x: (x['business_id'], 1)) \
                         .groupByKey() \
                         .mapValues(lambda x: len(x))
    user_tip_cnt = tip_rdd.map(lambda row: json.loads(row)) \
                          .map(lambda x: (x['user_id'], 1)) \
                          .groupByKey() \
                          .mapValues(lambda x: len(x))
    ### Checkin
    checkin_rdd = sc.textFile(checkin_filepath)
    checkin_cnt = checkin_rdd.map(lambda row: json.loads(row)) \
                             .map(lambda x: (x['business_id'], sum(x['time'].values()))) \
                             .reduceByKey(lambda a, b: a + b)

    train_rdd = sc.textFile(train_file_name)
    header1 = train_rdd.first()
    train = train_rdd.filter(lambda x: x != header1) \
                     .map(lambda x: x.split(',')) \
                     .map(lambda x: (x[0], (x[1], float(x[2]))))

    test_rdd = sc.textFile(test_file_name)
    header2 = test_rdd.first()
    test = test_rdd.filter(lambda x: x != header2) \
                   .map(lambda x: x.split(',')) \
                   .map(lambda x: (x[0], (x[1], 0)))

    ### Model based
    # Prepare training data
    user_table = train.join(user_info) \
                      .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][1][0], x[1][1][1], x[1][1][2]))) \
                      .leftOuterJoin(user_tip_cnt) \
                      .map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1])))

    bus_table = business_info.leftOuterJoin(photo_cnt) \
                             .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1]))) \
                             .leftOuterJoin(bus_tip_cnt) \
                             .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1]))) \
                             .leftOuterJoin(checkin_cnt) \
                             .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][1])))

    full_table = user_table.join(bus_table) \
                           .map(lambda x: (
                            x[1][0][0], x[0], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][1][0], x[1][1][1], x[1][1][2],
                            x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][0][1])) \
                           .collect()

    cols = ['user_id', 'business_id', 'user_review_cnt', 'user_stars', 'useful', 'user_tip_cnt', 'bus_stars',
            'bus_review_cnt', 'is_restaurant', 'state', 'photo_cnt', 'tip_cnt', 'checkin_cnt', 'rating']

    df = pd.DataFrame(full_table, columns=cols)
    df = pd.get_dummies(df, columns=['state'], drop_first=True)
    df = df[[col for col in df.columns if col != 'rating'] + ['rating']]
    df = df.fillna(0)

    x = df.iloc[:, 2:-1]
    y = df['rating']

    # Train the xgboost model
    xgbr = xgb.XGBRegressor(verbosity=0)
    xgbr.fit(x, y)

    # Prepare test data
    user_table1 = test.join(user_info) \
                      .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][1][0], x[1][1][1], x[1][1][2]))) \
                      .leftOuterJoin(user_tip_cnt) \
                      .map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1])))

    full_table1 = user_table1.join(bus_table) \
                             .map(lambda x: (
                              x[1][0][0], x[0], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][1][0], x[1][1][1], x[1][1][2],
                              x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][0][1])) \
                             .collect()

    MB_pred_df = pd.DataFrame(full_table1, columns=cols)
    MB_pred_df = pd.get_dummies(MB_pred_df, columns=['state'], drop_first=True)
    MB_pred_df = MB_pred_df[[col for col in MB_pred_df.columns if col != 'rating'] + ['rating']]
    MB_pred_df = MB_pred_df.fillna(0)

    x_test = MB_pred_df.iloc[:, 2:-1]

    # Make prediction
    MB_pred_df['pred'] = xgbr.predict(x_test)

    MB_pred_dict = {(record['user_id'], record['business_id']): (record['pred'], record['user_review_count'], record['bus_review_count']) for record in MB_pred_df.to_dict('record')}

    ### Collaborative Filtering
    train = train_rdd.filter(lambda x: x != header1) \
        .map(lambda x: x.split(',')) \
        .map(lambda x: (x[0], x[1], float(x[2])))

    record_dict = train.map(lambda x: ((x[0], x[1]), x[2])) \
        .collectAsMap()

    # reserve business w/ 30+ users -> eventually will calculate similarity only for business co-rated by 30+ users
    bus_user = train.map(lambda x: (x[1], x[0])) \
                    .groupByKey() \
                    .mapValues(lambda x: set(x)) \
                    .filter(lambda x: len(x[1]) >= 30) \
                    .sortByKey()

    bus_user_dict = bus_user.collectAsMap()

    user_bus_dict = train.map(lambda x: (x[0], x[1])) \
                         .groupByKey() \
                         .mapValues(lambda x: set(x)) \
                         .collectAsMap()

    # find all pairs with #of co-rating users >= 30
    pairs = bus_user.keys().cartesian(bus_user.keys()) \
                    .filter(lambda x: x[0] < x[1]) \
                    .map(lambda x: ((x[0], x[1]), bus_user_dict[x[0]].intersection(bus_user_dict[x[1]]))) \
                    .filter(lambda x: len(x[1]) >= 30)

    # calculate pearson correlation for each pair: {(I1, I2): weight}
    similarity = pairs.map(lambda x: cal_pearson_sim(x[0], x[1], record_dict)) \
                      .collectAsMap()

    # Make final prediction
    predictions =test.map(lambda x: make_final_pred(x[0], x[1], MB_pred_dict, similarity, user_bus_dict, record_dict)) \
                     .collect()

    output = 'user_id, business_id, prediction'
    for i in predictions:
        output += ('\n' + i[0] + ',' + i[1] + ',' + str(i[2]))

    with open(output_file_name, 'w+') as file:
        file.write(output)
    file.close()

    end = time.time()
    print(end - begin)
