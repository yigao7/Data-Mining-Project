# Collaborative filtering (item-based) recommendation system with Pearson similarity

from pyspark import SparkContext
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

def make_pred(u, i, w, user_bus_dict, r, bus_avg_r):
    if u not in user_bus_dict:   # cold start
        if i in bus_avg_r:
            pred = bus_avg_r[i]
        else:
            pred = 3
    else:
        N = set()
        for n in user_bus_dict[u]:
            if tuple(sorted((i,n))) in w and w[tuple(sorted((i,n)))] > 0:
                N.add(n)
        if N == set():      # no weight calculated, or no similar businesses, to form neighborhood
            if i in bus_avg_r:
                pred = bus_avg_r[i]
            else:
                pred = 3
        else:
            numerator = sum([r[(u,n)]*w[tuple(sorted((i,n)))] for n in N])
            denominator = sum([abs(w[tuple(sorted((i,n)))]) for n in N])
            pred = numerator / denominator
    return (u, i, pred)

if __name__ == '__main__':

    sc = SparkContext('local[*]', 'task1')

    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    rdd1 = sc.textFile(train_file_name)
    header1 = rdd1.first()
    train = rdd1.filter(lambda x: x != header1) \
                .map(lambda x: x.split(',')) \
                .map(lambda x: (x[0], x[1], float(x[2])))

    rdd2 = sc.textFile(test_file_name)
    header2 = rdd2.first()
    test = rdd2.filter(lambda x: x != header2) \
                .map(lambda x: x.split(',')) \
                .map(lambda x: (x[0], x[1], float(x[2])))

    record_dict = train.map(lambda x: ((x[0], x[1]), x[2])) \
                        .collectAsMap()

    # calculate average rating for each business in record
    bus_avg_r =  train.map(lambda x: (x[1], x[2])) \
                      .groupByKey() \
                      .mapValues(lambda x: sum(x) / len(x)) \
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

    # make prediction: (user_id, bus_id, prediction)
    predicitons = test.map(lambda x: make_pred(x[0], x[1], similarity, user_bus_dict, record_dict, bus_avg_r)) \
                      .collect()

    output = 'user_id, business_id, prediction'
    for i in predicitons:
        output += ('\n' + i[0] + ',' + i[1] + ',' + str(i[2]))

    with open(output_file_name, 'w+') as file:
        file.write(output)
    file.close()
