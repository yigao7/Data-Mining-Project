# Train XGBregressor to get feature-based model.

from pyspark import SparkContext
import json
import pandas as pd
import xgboost as xgb
from math import sqrt
import sys

def predict(u, i, truth, df2):
    pred = df2.loc[(df2['user_id'] == u) & (df2['business_id'] == i),'pred'].iloc[0]
    return(u, i, pred, truth)

if __name__ == '__main__':

    sc = SparkContext('local[*]', 'task2.2')
    sc.setLogLevel("ERROR")

    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    business_filepath = folder_path + "/business.json"
    user_filepath = folder_path + "/user.json"
    train_file_name = folder_path + "/yelp_train.csv"

    business_rdd = sc.textFile(business_filepath)
    business_info = business_rdd.map(lambda row: json.loads(row)) \
        .map(lambda x: (x['business_id'], (x['stars'], x['review_count'])))

    user_rdd = sc.textFile(user_filepath)
    user_info = user_rdd.map(lambda row: json.loads(row)) \
        .map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'])))

    train_rdd = sc.textFile(train_file_name)
    header1 = train_rdd.first()
    train = train_rdd.filter(lambda x: x != header1) \
        .map(lambda x: x.split(',')) \
        .map(lambda x: (x[0], (x[1], float(x[2]))))

    test_rdd = sc.textFile(test_file_name)
    header2 = test_rdd.first()
    test = test_rdd.filter(lambda x: x != header2) \
                   .map(lambda x: x.split(',')) \
                   .map(lambda x: (x[0], x[1]))

    # Prepare training data
    train_user_info = train.join(user_info) \
                           .map(lambda x: (x[1][0][0], (x[0], x[1][1][0], x[1][1][1], x[1][0][1])))

    train_all_info = train_user_info.join(business_info) \
        .map(lambda x: (x[1][0][0], x[0], x[1][0][1], x[1][0][2], x[1][1][1], x[1][1][0], x[1][0][3])) \
        .collect()

    cols = ['user_id', 'business_id', 'user_review_count', 'user_stars', 'bus_review_count', 'bus_stars', 'rating']
    df = pd.DataFrame(train_all_info, columns=cols)
    x = df.iloc[:, 2:6]
    y = df['rating']

    # Train the xgboost model
    xgbr = xgb.XGBRegressor(verbosity=0)
    xgbr.fit(x, y)

    # Prepare test data
    test_user_info = test.join(user_info) \
                         .map(lambda x: (x[1][0], (x[0], x[1][1][0], x[1][1][1])))

    test_all_info = test_user_info.join(business_info) \
        .map(lambda x: (x[1][0][0], x[0], x[1][0][1], x[1][0][2], x[1][1][1], x[1][1][0])) \
        .collect()

    cols2 = ['user_id', 'business_id', 'user_review_count', 'user_stars', 'bus_review_count', 'bus_stars']
    df2 = pd.DataFrame(test_all_info, columns=cols2)
    x_test = df2.iloc[:, 2:6]

    # Make prediction
    df2['pred'] = xgbr.predict(x_test)

    predicted_test = test_rdd.filter(lambda x: x != header2) \
                               .map(lambda x: x.split(',')) \
                               .map(lambda x: (x[0], x[1], float(x[2]))) \
                               .map(lambda x: predict(x[0], x[1], x[2], df2)) \
                               .collect()

    RMSE = sqrt(sum([(predicted_test[i][3] - predicted_test[i][2]) ** 2 for i in range(len(predicted_test))]) / len(predicted_test))
    print(RMSE)
