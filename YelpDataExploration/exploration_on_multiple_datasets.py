#  explore two datasets together containing review information (test_review.json) and business information (business.json) 
#  to find a) What is the average stars for each city? b) compare the execution time of using two methods to print top 10 cities with highest
#  stars

from pyspark import SparkContext
import json
import time
import sys

if __name__ == '__main__':
    sc = SparkContext('local[*]', 'task3')

    review_filepath = sys.argv[1]
    business_filepath = sys.argv[2]
    output_filepath_question_a = sys.argv[3]
    output_filepath_question_b = sys.argv[4]

    reivew = sc.textFile(review_filepath)
    business = sc.textFile(business_filepath)
    review_star = reivew.map(lambda row: json.loads(row)).map(lambda x: (x['business_id'], x['stars']))
    business_city = business.map(lambda row: json.loads(row)).map(lambda x: (x['business_id'], x['city']))

    # A. What is the average stars for each city? 
    city_avg_review = review_star.leftOuterJoin(business_city) \
                             .map(lambda x: (x[1][1], x[1][0])) \
                             .groupByKey() \
                             .mapValues(list) \
                             .mapValues(lambda x: sum(x)/len(x)) \
                             .sortBy(lambda x: (-x[1], x[0])) \
                             .collect()

    # print(city_avg_review)

    with open(output_filepath_question_a, 'w+') as output_a:
        output_a.write('city,stars\n')
        for kv in city_avg_review:
            output_a.write(kv[0]+','+str(kv[1])+'\n')
    output_a.close()

    # B. Compare the execution time of using two methods to print top 10 cities with highest stars.
    #    Method1: Collect all the data, sort in python, and then print the first 10 cities
    #    Method2: Sort in Spark, take the first 10 cities, and then print these 10 cities

    output_3b = {}

    # M1
    m1_start_time = time.clock()
    with open(review_filepath, 'r') as file:
        review_file = file.readlines()
    review_list = []
    for line in review_file:
        line = json.loads(line)
        review_list.append((line['business_id'], line['stars']))
    # print(review_list)
    with open(business_filepath, 'r') as file:
        bus_file = file.readlines()
    bus_list = []
    for line in bus_file:
        line = json.loads(line)
        bus_list.append((line['business_id'], line['city']))

    merge_dict = {}
    for i in review_list:
        if i[0] in merge_dict:
            merge_dict[i[0]]['stars'].append(i[1])
        else:
            merge_dict[i[0]] = {}
            merge_dict[i[0]]['stars'] = [i[1]]
    for i in bus_list:
        if i[0] in merge_dict:
            merge_dict[i[0]]['city'] = i[1]
    # print(merge_dict)

    city_dict = {}
    for i in merge_dict:
        if merge_dict[i]['city'] not in city_dict:
            city_dict[merge_dict[i]['city']] = merge_dict[i]['stars']
        else:
            city_dict[merge_dict[i]['city']] += merge_dict[i]['stars']
    city_avg_dict = {k: sum(v)/len(v) for k,v in city_dict.items()}
    city_avg_dict = sorted(city_avg_dict.items(), key=lambda x: (-x[1], x[0]))

    result_m1 = json.dumps(city_avg_dict[:10])

    print(result_m1)
    m1_end_time = time.clock()
    m1_time = m1_end_time - m1_start_time
    print(m1_time)

    # M2
    m2_start_time = time.clock()
    reivew = sc.textFile(review_filepath)
    business = sc.textFile(business_filepath)
    review_star = reivew.map(lambda row: json.loads(row)).map(lambda x: (x['business_id'], x['stars']))
    business_city = business.map(lambda row: json.loads(row)).map(lambda x: (x['business_id'], x['city']))

    result_m2 = review_star.leftOuterJoin(business_city) \
        .map(lambda x: (x[1][1], x[1][0])) \
        .groupByKey() \
        .mapValues(list) \
        .mapValues(lambda x: sum(x) / len(x)) \
        .sortBy(lambda x: (-x[1], x[0])) \
        .take(10)

    print(result_m2)
    m2_end_time = time.clock()
    m2_time = m2_end_time - m2_start_time
    print(m2_time)

    output_3b['m1'],  output_3b['m2']= m1_time, m2_time
    output_3b['reason'] = "m2 is much faster than m1. Executing data reading, mapping, joining and grouping using python requires processing the entire datasets all at once, while spark uses RDD and processes them in parallel. In addition, spark utilizes more memory, avoiding reading from and writing to disks."

    output_3b_json = json.dumps(output_3b)

    with open(output_filepath_question_b, 'w+') as output_b:
        output_b.write(output_3b_json)
    output_b.close()

