# Construct a social network graph. Assume that each node is uniquely labeled, and that links are undirected and unweighted.
# Each node represents a user. There should be an edge between two nodes if the number of common businesses reviewed by two users is greater than or equivalent to the filter threshold.
# Then explore the Spark GraphFrames library to detect communities in the network graph.

from pyspark import SparkContext
from pyspark.sql import SparkSession
import sys
from graphframes import GraphFrame
from itertools import combinations
import time

if __name__ == '__main__':

    start = time.time()

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")

    threshold = int(sys.argv[1])
    input_filepath = sys.argv[2]
    community_output_file_path = sys.argv[3]

    rdd = sc.textFile(input_filepath)
    header = rdd.first()
    record = rdd.filter(lambda x: x != header)\
                .map(lambda x: x.split(','))

    user_bus = record.groupByKey() \
                     .mapValues(lambda x: (set(x))) \
                     .sortBy(lambda x: x[0]) \
                     .collectAsMap()

    pairs = list(combinations(user_bus.keys(),2))

    edges = []
    vertices = []
    for pair in pairs:
        i, j = pair[0], pair[1]
        if len(user_bus[i] & user_bus[j]) >= threshold:
            edges.append(pair)
            vertices.append(i)
            vertices.append(j)
    vertices = [[i] for i in list(set(vertices))]

    spark = SparkSession.builder.getOrCreate()

    vertices_df = spark.createDataFrame(vertices, ["id"])
    edges_df = spark.createDataFrame(edges, ["src", "dst"])

    g = GraphFrame(vertices_df, edges_df)
    communities = g.labelPropagation(maxIter=5).rdd

    result = communities.map(lambda x: (x[1],x[0])) \
                        .groupByKey() \
                        .map(lambda x: (x[0], sorted(list(x[1])))) \
                        .sortBy(lambda x: (len(x[1]), x[1])) \
                        .collect()

    output = ', '.join(result[0][1])
    for i in result[1:]:
        output += '\n'
        output += ', '.join(i[1])

    with open(community_output_file_path, 'w+') as file:
        file.write(output)
    file.close()
