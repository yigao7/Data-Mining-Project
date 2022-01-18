# Explore the Ta Feng dataset to find the frequent itemsets. Data is from Kaggle (https://bit.ly/2miWqFS).
# Find product IDs associated with a given customer ID each day. Aggregate all purchases a customer makes within a day into one basket. 
# In other words, assume a customer purchases at once all items purchased within a day.

from pyspark import SparkContext
import sys
from itertools import combinations
import time

def get_new_candidates(current_candidates, num_item):
    new_candidates = []
    for i in range(len(current_candidates)-1):
        for j in range(i+1, len(current_candidates)):
            if current_candidates[i][:-1] != current_candidates[j][:-1]:
                break
            else:
                candidate = tuple(sorted((set(current_candidates[i]).union(set(current_candidates[j])))))
                if set(combinations(candidate, num_item-1)).issubset(set(current_candidates)):
                    new_candidates.append(candidate)
    return new_candidates

def apriori(partition, whole_file_size, s):
    candidates_except_singles = []
    counter = {}
    partition = list(partition)
    subset_support = len(partition) / whole_file_size * s
    for basket in partition:
        for item in basket[1]:
            if item in counter:
                counter[item] += 1
            else:
                counter[item] = 1
    singles = [key for key, value in counter.items() if value >= subset_support]
    if singles != []:
        counter = {}
        for basket in partition:
            combos = list(combinations(basket[1], 2))
            for combo in combos:
                combo = tuple(sorted(combo))
                if set(combo).issubset(set(singles)):
                    if combo in counter:
                        counter[combo] += 1
                    else:
                        counter[combo] = 1
        current_candidates = sorted([key for key, value in counter.items() if value >= subset_support])
        candidates_except_singles += current_candidates
    num_item = 2
    while current_candidates != []:
        num_item += 1
        counter = {}
        new_candidates = get_new_candidates(current_candidates, num_item)
        for basket in partition:
            for candidate in new_candidates:
                if set(candidate).issubset(set(basket[1])):
                    if candidate in counter:
                        counter[candidate] += 1
                    else:
                        counter[candidate] = 1
        current_candidates = sorted([key for key, value in counter.items() if value >= subset_support])
        candidates_except_singles += current_candidates
    return [(item,) for item in singles] + candidates_except_singles

def count_all_baskets(basket, candidates):
    counter = {}
    for i in candidates:
        if len(i) == 1:
            if i[0] in set(basket[1]):
                if i in counter:
                    counter[i] += 1
                else:
                    counter[i] = 1
        else:
            if set(i).issubset(set(basket[1])):
                if i in counter:
                    counter[i] += 1
                else:
                    counter[i] = 1
    return [(k, v) for k, v in counter.items()]

if __name__ == '__main__':

    start = time.time()

    sc = SparkContext('local[*]', 'task2')

    k = int(sys.argv[1])
    s = int(sys.argv[2])
    input_filepath = sys.argv[3]
    output_filepath = sys.argv[4]

    # Step 1: Data preprocessing
    rdd = sc.textFile(input_filepath).map(lambda x: x.replace('"', ''))\
                                     .map(lambda x: list(x.split(',')))
    header = rdd.first()
    record = rdd.filter(lambda x: x != header)\
                .map(lambda x: (x[0]+'-'+x[1], int(x[5])))\

    '''
    with open(preprocessed_filepath, 'w+') as output:
        output.write('DATE-CUSTOMER_ID,PRODUCT_ID\n')
        for kv in record.collect():
            output.write(kv[0]+','+str(kv[1])+'\n')
    output.close()
    '''

    # Step 2: Apply SON

    basket = record.groupByKey() \
                   .mapValues(lambda x: (list(x))) \
                   .filter(lambda x: len(x[1]) > k)
    num_baskets = basket.count()

    # SON Phase 1
    candidates_after_son_1 = basket.mapPartitions(lambda partition: apriori(partition, num_baskets, s)) \
        .distinct() \
        .sortBy(lambda x: (len(x), str(x))) \
        .collect()

    output = 'Candidates:\n'
    current_len = 1
    for candidate in candidates_after_son_1:
        if len(candidate) == current_len:
            output += '('
            index = 0
            while index < len(candidate) - 1:
                output += "'" + str(candidate[index]) + "', "
                index += 1
            output += "'" + str(candidate[index]) + "'),"
        else:
            current_len += 1
            output = output[:-1] + '\n\n('
            index = 0
            while index < len(candidate) - 1:
                output += "'" + str(candidate[index]) + "', "
                index += 1
            output += "'" + str(candidate[index]) + "'"
            output += '),'
    output = output[:-1] + '\n\nFrequent Itemsets:\n'

    # SON Phase 2
    truly_frequent_sets = basket.flatMap(lambda x: count_all_baskets(x, candidates_after_son_1)) \
        .reduceByKey(lambda a, b: a + b) \
        .filter(lambda x: x[1] >= s) \
        .map(lambda x: x[0]) \
        .sortBy(lambda x: (len(x), str(x))) \
        .collect()

    print(truly_frequent_sets)

    current_len = 1
    for set in truly_frequent_sets:
        if len(set) == current_len:
            output += '('
            index = 0
            while index < len(set) - 1:
                output += "'" + str(set[index]) + "', "
                index += 1
            output += "'" + str(set[index]) + "'),"
        else:
            current_len += 1
            output = output[:-1] + '\n\n('
            index = 0
            while index < len(set) - 1:
                output += "'" + str(set[index]) + "', "
                index += 1
            output += "'" + str(set[index]) + "'"
            output += '),'
    output = output[:-1]

    with open(output_filepath, 'w+') as output_file:
        output_file.write(output)
    output_file.close()

    print('Duration: ', time.time() - start)
