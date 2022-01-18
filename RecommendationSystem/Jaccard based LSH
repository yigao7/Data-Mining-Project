# Implement the Locality Sensitive Hashing algorithm with Jaccard similarity using Yelp data.
# Focus on the “0 or 1” ratings rather than the actual ratings/stars from the users. Specifically, if a user has rated a business, the user’s contribution in the characteristic matrix is 1. If the user hasn’t rated the business, the contribution is 0. I will identify similar businesses whose similarity >= 0.5.

from pyspark import SparkContext
import random
import itertools
import sys

def hash_index_and_compute_sig(x, num_hash, a_list, b_list, m):
    signature = []
    for i in range(num_hash):
        a = a_list[i]
        b = b_list[i]
        new_indices = [(a*index+b)%m for index in x]
        signature.append(min(new_indices))
    return signature

def partition_to_bands(b, x, num_hash):
    # b = # of bands, x = (business, signature of a business)
    output = []
    r = num_hash/b
    for i in range(b):
        output.append((str((i+1, x[1][int(i*r):int(i*r+r)])), x[0]))
    return output

def find_original_jaccard(x, bus_user):
    # x = pair, bus_user = {business: [users]}
    jaccard_similarity = len(bus_user[x[0]].intersection(bus_user[x[1]])) / len(bus_user[x[0]].union(bus_user[x[1]]))
    return (x, jaccard_similarity)

if __name__ == '__main__':

    sc = SparkContext('local[*]', 'task1')

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    s = .5
    rdd = sc.textFile(input_file_name)
    header = rdd.first()
    record = rdd.filter(lambda x: x != header) \
                .map(lambda x: x.split(','))

    bus_user = record.map(lambda x: (x[1], x[0])) \
                     .groupByKey() \
                     .mapValues(lambda x: set(x)) \
                     .collectAsMap()

    user_index = record.map(lambda x: (x[0])) \
                       .distinct() \
                       .sortBy(lambda x: x) \
                       .zipWithIndex() \
                       .collectAsMap()

    # generate hash function parameters
    num_hash = 80
    a_list = [random.randint(1, 10001) for i in range(num_hash)]
    b_list = [random.randint(0, 10000) for i in range(num_hash)]
    m = 10000

    sig_matrix = record.map(lambda x: (x[1], user_index[x[0]])) \
                       .groupByKey() \
                       .mapValues(lambda x: (sorted(list(set(x))))) \
                       .sortBy(lambda x: x[0]) \
                       .mapValues(lambda x: hash_index_and_compute_sig(x, num_hash, a_list, b_list, m))

    candidates = sig_matrix.flatMap(lambda x: partition_to_bands(40, x, num_hash)) \
                           .groupByKey() \
                           .mapValues(lambda x: list(x)) \
                           .filter(lambda x: len(x[1]) > 1) \
                           .flatMap(lambda x: [combo for combo in itertools.combinations(x[1], 2)]) \
                           .distinct()

    final_pairs = candidates.map(lambda x: find_original_jaccard(x, bus_user)) \
                            .filter(lambda x: x[1] >= s) \
                            .sortBy(lambda x: x[0]) \
                            .collect()

    output = 'business_id_1, business_id_2, similarity'
    for i in final_pairs:
        output += ('\n'+i[0][0]+','+i[0][1]+','+str(i[1]))

    with open(output_file_name, 'w+') as file:
        file.write(output)
    file.close()

    print(candidates.getNumPartitions())
