# Since processing large volumes of data requires performance decisions, properly partitioning the data for processing is imperative.
# I use a customized partition function to improve the performance of map and reduce tasks. A time duration comparison between the 
# default partition and the customized partition (RDD built using the partition function) is displayed.

from pyspark import SparkContext
import json
import random
import sys
import time

if __name__ == '__main__':
    sc = SparkContext('local[*]', 'task2')

    review_filepath = "review_filepath"
    output_filepath = "output_filepath"
    n_partition = 2

    rdd = sc.textFile(review_filepath)
    row = rdd.map(lambda row: json.loads(row))
    result = {'default': {}, 'customized': {}}


    def partition_func(x):
        return random.choice(range(int(n_partition)))


    bus_id = row.map(lambda x: (x['business_id'], 1))
    bus_id_customized = bus_id.partitionBy(int(n_partition), partition_func)

    default_start_time = time.clock()
    top10_bus = bus_id.reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda x: (-x[1], x[0])) \
        .take(10)
    default_end_time = time.clock()

    cus_start_time = time.clock()
    top10_bus_cus = bus_id_customized.reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda x: (-x[1], x[0])) \
        .take(10)
    cus_end_time = time.clock()

    result['default']['n_partition'] = bus_id.getNumPartitions()
    result['default']['n_items'] = bus_id.glom().map(lambda x: len(x)).collect()
    result['default']['exe_time'] = default_end_time - default_start_time
    result['customized']['n_partition'] = bus_id_customized.getNumPartitions()
    result['customized']['n_items'] = bus_id_customized.glom().map(lambda x: len(x)).collect()
    result['customized']['exe_time'] = cus_end_time - cus_start_time

    # Wrapping up
    result_json = json.dumps(result)

    print(result_json)

    with open(output_filepath, 'w+') as output:
        output.write(result_json)
    output.close()
