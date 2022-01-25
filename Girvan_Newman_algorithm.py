# Implemen the Girvan-Newman algorithm to detect the communities in the network graph
# 1. Calculate the betweenness of each edge.
# 2. Divide the graph into suitable communities, which reaches the global highest modularity. 

from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict
from operator import add
import random
import sys


def girvan_newman(root, neighbors):  # neighbors = {node1: {node2, node3, ...}}
    ### construct tree
    tree = dict()  # tree = {0: {node 1}, 1: {node2, node3}, ...}
    tree[0] = {root}
    upper_levels = {root}  # keep track of nodes in upper levels
    next_layer = neighbors[root]
    level = 0

    while len(next_layer) > 0:
        level += 1
        current_layer = next_layer
        tree[level] = current_layer
        next_layer = set()
        for node in current_layer:
            next_layer = next_layer.union(neighbors[node])
        upper_levels = upper_levels.union(current_layer)
        next_layer = next_layer - upper_levels

    ### assign credit
    edge_credit = defaultdict(int)
    node_credit = defaultdict(int)
    for node in tree[level]:  # leaves, assign credits of 1
        node_credit[node] = 1
    last_layer = tree[level]

    while level > 1:
        level = level - 1
        current_layer = tree[level]
        next_layer = tree[level - 1]
        parent_path_num = defaultdict(int)
        for node2 in current_layer:
            node_credit[node2] += 1
            for node3 in next_layer:
                if node3 in neighbors[node2]:
                    parent_path_num[node2] += 1
        for node1 in last_layer:
            parents = []
            for node2 in current_layer:
                if node2 in neighbors[node1]:
                    parents.append(node2)
            for parent in parents:
                edge_credit[tuple(sorted((node1, parent)))] = node_credit[node1] * (parent_path_num[parent] / sum([parent_path_num[i] for i in parents]))
                node_credit[parent] += node_credit[node1] * (parent_path_num[parent] / sum([parent_path_num[i] for i in parents]))
        last_layer = current_layer

    for node1 in last_layer:
        edge_credit[tuple(sorted((node1, root)))] = node_credit[node1]

    return [(k, v) for k, v in edge_credit.items()]


def get_communities(vertices, neighbors_dict):
    communities = []
    unvisited = set(vertices)

    while len(unvisited) > 0:
        root = random.choice(list(unvisited))
        community = {root}
        unvisited.remove(root)
        neighbors = neighbors_dict[root]
        parents = {root}

        while len(neighbors) > 0:
            unvisited = unvisited - neighbors
            parents = parents.union(neighbors)
            community = community.union(neighbors)
            next_neighbors = set()
            for neighbor in neighbors:
                next_neighbors = next_neighbors.union(neighbors_dict[neighbor]).difference(parents)
            neighbors = next_neighbors

        communities.append(community)

    return communities

def cal_modularity(m, A, S, K):
    Q = 0
    for s in S:
        for i in s:
            for j in s:
                Q += A[(i,j)] - K[i]*K[j] / (2*m)
    Q = Q / (2*m)
    return Q

if __name__ == '__main__':

    sc = SparkContext('local[*]', 'task2')
    sc.setLogLevel("WARN")

    threshold = int(sys.argv[1])
    input_filepath = sys.argv[2]
    betweenness_output = sys.argv[3]
    community_output = sys.argv[4]

    rdd = sc.textFile(input_filepath)
    header = rdd.first()
    record = rdd.filter(lambda x: x != header)\
                .map(lambda x: x.split(','))

    user_bus = record.groupByKey() \
                     .mapValues(lambda x: (set(x))) \
                     .sortBy(lambda x: x[0]) \
                     .collectAsMap()

    pairs = list(combinations(user_bus.keys(), 2))

    edges = []
    vertices = []
    for pair in pairs:
        i, j = pair[0], pair[1]
        if len(user_bus[i] & user_bus[j]) >= threshold:
            edges.append((i, j))
            edges.append((j, i))
            vertices.append(i)
            vertices.append(j)
    vertices = [[i] for i in list(set(vertices))]

    # Matrix A
    neighbors = sc.parallelize(edges) \
                  .groupByKey() \
                  .mapValues(lambda x: (set(x))) \
                  .collectAsMap()

    betweenness = sc.parallelize(vertices) \
        .flatMap(lambda x: girvan_newman(x[0], neighbors)) \
        .reduceByKey(add) \
        .map(lambda x: (x[0], x[1] / 2)) \
        .sortBy(lambda x: (-x[1], x[0])) \
        .collect()

    output = str(betweenness[0][0]) + ',' + str(round(betweenness[0][1], 5))

    for i in betweenness[1:]:
        output += '\n' + str(i[0]) + ',' + str(round(i[1], 5))

    with open(betweenness_output, 'w+') as file:
        file.write(output)
    file.close()

    # Degree matrix
    K = sc.parallelize(edges) \
          .groupByKey() \
          .mapValues(lambda x: len(set(x))) \
          .collectAsMap()

    # m to calculate modularity
    m = len(edges) / 2

    # Construct A matrix for modularity
    A_matrix = {}
    for i in [x[0] for x in vertices]:
        for j in [x[0] for x in vertices]:
            if j in neighbors[i]:
                A_matrix[(i,j)] = 1
            else:
                A_matrix[(i,j)] = 0

    max_modularity = -1
    best_communities = None

    for i in range(int(m)):
        betweenness = sc.parallelize(vertices) \
                        .flatMap(lambda x: girvan_newman(x[0], neighbors)) \
                        .reduceByKey(add) \
                        .map(lambda x: (x[0], x[1] / 2)) \
                        .collectAsMap()

        edges_to_cut = [k for k, v in betweenness.items() if v == max(betweenness.values())]

        for edge in edges_to_cut:
            if edge[0] != edge[1]:
                neighbors[edge[0]].remove(edge[1])
                neighbors[edge[1]].remove(edge[0])

        communities = get_communities([x[0] for x in vertices], neighbors)

        modularity = cal_modularity(m, A_matrix, communities, K)

        if modularity > max_modularity:
            max_modularity = modularity
            best_communities = communities

    communities = sc.parallelize(best_communities) \
                   .map(lambda x: sorted(x)) \
                   .sortBy(lambda x: (len(x), x)) \
                   .collect()

    output = ''
    for i in communities[0]:
        output += "'" + i + "', "

    for i in communities[1:]:
        output = output[:-2]
        output += '\n'
        for j in i:
            output += "'" + j + "', "
    output = output[:-2]

    with open(community_output_file, 'w+') as file:
        file.write(output)
    file.close()
