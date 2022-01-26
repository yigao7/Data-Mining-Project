from pyspark import SparkContext
import sys
import random
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from math import sqrt
from itertools import combinations, product

def Mahalanobis_Distance(X, cluster_stat):
    # X = [x1,...xd], cluster_stat = {'N': int, 'SUM': array, 'SUMSQ': array}
    C = cluster_stat['SUM'] / cluster_stat['N']     # centroid
    V = cluster_stat['SUMSQ'] / cluster_stat['N'] - (cluster_stat['SUM'] / cluster_stat['N'])**2    # cluster variance
    SD = V**.5
    Y = (X-C)/SD
    distance = sqrt(sum([y**2 for y in Y]))
    return distance

def assign_to_cluster(index, X, clusters, d):
    min_distance = np.inf
    for cluster in clusters:
        distance = Mahalanobis_Distance(X, clusters[cluster])
        if distance < min_distance:
            nearest_cluster = cluster
            min_distance = distance
    if min_distance < 2*sqrt(d):
        return (index, nearest_cluster, X)
    else:
        return (index, -1, X)

def cluster_Mahalanobis_Distance(stat1, stat2):
    # stat1, stat2 = {'N': int, 'SUM': array, 'SUMSQ': array}
    C1 = stat1['SUM'] / stat1['N']
    C2 = stat2['SUM'] / stat2['N']
    V = (stat1['SUMSQ']+stat2['SUMSQ']) / (stat1['N']+stat2['N']) - ((stat1['SUM']+stat2['SUM']) / (stat1['N']+stat2['N']))**2    # cluster variance
    SD = V**.5
    Y = (C1-C2)/SD
    distance = sqrt(sum([y**2 for y in Y]))
    return distance

def output_cluster(index, DS_dict, CS_dict):
    cluster = -1
    for key in DS_dict:
        if index in key:
            cluster = DS_dict[key]
            break
    if cluster == -1:
        for key in CS_dict:
            if index in key:
                cluster = CS_dict[key]
                break
    return (index, cluster)

if __name__ == '__main__':
    sc = SparkContext('local[*]', 'task')
    sc.setLogLevel("ERROR")

    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    random.seed(553)

    rdd = sc.textFile(input_file)

    record = rdd.map(lambda x: x.split(',')) \
                .map(lambda x: ([int(i) for i in x[:2]] + [float(i) for i in x[2:]]))

    data = record.collect()
    random.shuffle(data)
    total_points = record.count()
    d = len(data[0][2:]) # num of dimensions

    # Step 1: Load 20% of the data randomly.
    first_round_data = np.array(data[:int(total_points/5)])

    index_dict = {x[0]: i for i, x in enumerate(first_round_data)}

    # Step 2: Run K-Means with a large K.
    kmeans = KMeans(n_clusters=10*n_cluster, random_state=0).fit(first_round_data[:, 2:])

    # Step 3: Move all the clusters that contain only one point to RS (outliers).
    clusters = defaultdict(list)
    clustered_pts = list(zip(kmeans.labels_, first_round_data))
    for x in clustered_pts:
        clusters[x[0]].append(int(x[1][0]))
    RS = []
    for cluster in clusters:
        if len(clusters[cluster]) == 1:
            RS += clusters[cluster]

    RS_data = first_round_data[[index_dict[x] for x in RS]]

    first_round_DS_data = np.delete(first_round_data, [index_dict[x] for x in RS], 0)

    # Step 4: Run K-Means again to cluster the rest of data points with K = the number of input clusters
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(first_round_DS_data[:, 2:])

    # Step 5: Use the K-Means result from Step 4 to generate the DS clusters
    # (i.e., discard their points and generate statistics).
    clustered_pts = list(zip(kmeans.labels_, first_round_DS_data))

    DS_dict = defaultdict(list)
    DS_stat = defaultdict(dict)

    for x in clustered_pts:
        DS_dict[x[0]].append(int(x[1][0]))
        if x[0] not in DS_stat:
            DS_stat[x[0]]['N'] = 1
            DS_stat[x[0]]['SUM'] = np.array(x[1][2:])
            DS_stat[x[0]]['SUMSQ'] = np.array([i**2 for i in x[1][2:]])
        else:
            DS_stat[x[0]]['N'] += 1
            DS_stat[x[0]]['SUM'] += np.array(x[1][2:])
            DS_stat[x[0]]['SUMSQ'] += np.array([i ** 2 for i in x[1][2:]])

    # Step 6: Run K-Means on the points in RS with a large K (e.g., 5 times of the number of the input clusters)
    # to generate CS (clusters with more than one points) and RS (clusters with only one point).
    kmeans = KMeans(n_clusters=int(max(1,0.8*len(RS))), random_state=0).fit(RS_data[:, 2:])
    clustered_pts = list(zip(kmeans.labels_, RS_data))

    clusters = defaultdict(list)
    for x in clustered_pts:
        clusters[x[0]].append(int(x[1][0]))

    RS,CS_cluster_labels = [],[]
    for cluster in clusters:
        if len(clusters[cluster]) == 1:
            RS += clusters[cluster]
        else:
            CS_cluster_labels.append(cluster)

    CS_dict = defaultdict(list)
    CS_stat = defaultdict(dict)
    label_dict = {}
    label = -1

    for x in clustered_pts:
        if x[0] in CS_cluster_labels:
            if x[0] not in label_dict:
                label += 1
                label_dict[x[0]] = label
            else:
                label = label_dict[x[0]]
            CS_dict[label].append(int(x[1][0]))
            if label not in CS_stat:
                CS_stat[label]['N'] = 1
                CS_stat[label]['SUM'] = np.array(x[1][2:])
                CS_stat[label]['SUMSQ'] = np.array([i ** 2 for i in x[1][2:]])
            else:
                CS_stat[label]['N'] += 1
                CS_stat[label]['SUM'] += np.array(x[1][2:])
                CS_stat[label]['SUMSQ'] += np.array([i ** 2 for i in x[1][2:]])

    # Output: “the number of the discard points”, “the number of the clusters in the compression set”,
    # “the number of the compression points”, and “the number of the points in the retained set”.
    num_DS = sum([len(value) for value in DS_dict.values()])
    num_cluster_CS = len(CS_dict)
    num_CS = sum([len(value) for value in CS_dict.values()])
    num_RS = len(RS)
    output = f"The intermediate results:\nRound 1: {num_DS},{num_cluster_CS},{num_CS},{num_RS}\n"

    for round in range(1,5):
        # Step 7: load another 20% of the data randomly
        if round < 4:
            this_round_data = data[round*int(total_points/5): (round+1)*int(total_points/5)]
        else:
            this_round_data = data[round * int(total_points / 5):]
        this_round_rdd = sc.parallelize(this_round_data)
        # Step 8: For the new points, compare them to each of the DS using the Mahalanobis Distance and assign
        # them to the nearest DS clusters if the distance is < 2√𝑑.
        new_DS = this_round_rdd.map(lambda x: assign_to_cluster(x[0], x[2:], DS_stat, d)) \
                               .filter(lambda x: x[1] != -1) \
                               .collect()

        not_DS = this_round_rdd.map(lambda x: assign_to_cluster(x[0], x[2:], DS_stat, d)) \
                               .filter(lambda x: x[1] == -1) \
                               .map(lambda x: x[0]) \
                               .collect()
        for point in new_DS:
            DS_dict[point[1]].append(int(point[0]))
            DS_stat[point[1]]['N'] += 1
            DS_stat[point[1]]['SUM'] += np.array(point[2])
            DS_stat[point[1]]['SUMSQ'] += np.array([x ** 2 for x in point[2]])

        # Step 10: For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
        new_RS = this_round_rdd.filter(lambda x: x[0] in not_DS) \
                               .map(lambda x: assign_to_cluster(x[0], x[2:], CS_stat, d)) \
                               .filter(lambda x: x[1] == -1) \
                               .map(lambda x: x[0]) \
                               .collect()
        RS += new_RS

        # Step 9: For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign
        # the points to the nearest CS clusters if the distance is < 2√𝑑
        new_CS = this_round_rdd.filter(lambda x: x[0] in not_DS) \
                               .map(lambda x: assign_to_cluster(x[0], x[2:], CS_stat, d)) \
                               .filter(lambda x: x[1] != -1) \
                               .collect()

        for point in new_CS:
            CS_dict[point[1]].append(int(point[0]))
            CS_stat[point[1]]['N'] += 1
            CS_stat[point[1]]['SUM'] += np.array(point[2])
            CS_stat[point[1]]['SUMSQ'] += np.array([x ** 2 for x in point[2]])

        # Step 11: Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters)
        # to generate CS (clusters with more than one points) and RS (clusters with only one point).
        RS_data = np.array(record.filter(lambda x: x[0] in RS).collect())

        kmeans = KMeans(n_clusters=int(max(1, 0.8 * len(RS_data))), random_state=0).fit(RS_data[:, 2:])
        clustered_pts = list(zip(kmeans.labels_, RS_data))

        clusters = defaultdict(list)
        for x in clustered_pts:
            clusters[x[0]].append(int(x[1][0]))

        RS, CS_cluster_labels = [], []
        for cluster in clusters:
            if len(clusters[cluster]) == 1:
                RS += clusters[cluster]
            else:
                CS_cluster_labels.append(cluster)

        label_dict = {}

        for x in clustered_pts:
            if x[0] in CS_cluster_labels:
                if x[0] not in label_dict:
                    label = max(CS_dict) + 1
                    label_dict[x[0]] = label
                else:
                    label = label_dict[x[0]]
                CS_dict[label].append(int(x[1][0]))
                if label not in CS_stat:
                    CS_stat[label]['N'] = 1
                    CS_stat[label]['SUM'] = np.array(x[1][2:])
                    CS_stat[label]['SUMSQ'] = np.array([i ** 2 for i in x[1][2:]])
                else:
                    CS_stat[label]['N'] += 1
                    CS_stat[label]['SUM'] += np.array(x[1][2:])
                    CS_stat[label]['SUMSQ'] += np.array([i ** 2 for i in x[1][2:]])

        # STEP 12: Merge CS clusters that have a Mahalanobis Distance < 2√𝑑.
        while True:
            num_clusters = len(CS_dict)
            pairs = list(combinations(list(CS_dict.keys()), 2))
            for pair in pairs:
                distance = cluster_Mahalanobis_Distance(CS_stat[pair[0]],CS_stat[pair[1]])
                if distance < 2*sqrt(d):
                    CS_stat[pair[0]]['N'] += CS_stat[pair[1]]['N']
                    CS_stat[pair[0]]['SUM'] += CS_stat[pair[1]]['SUM']
                    CS_stat[pair[0]]['SUMSQ'] += CS_stat[pair[1]]['SUMSQ']
                    CS_stat.pop(pair[1])
                    CS_dict[pair[0]] += CS_dict[pair[1]]
                    CS_dict.pop(pair[1])
                    break
            if num_clusters == len(CS_dict):    # No more combining
                break

        CS_dict = defaultdict(list, {i: CS_dict[k] for i, k in enumerate(sorted(CS_dict.keys()))})
        CS_stat = defaultdict(dict, {i: CS_stat[k] for i, k in enumerate(sorted(CS_stat.keys()))})

        # Step 13: In the last run, merge CS clusters with DS clusters that have a Mahalanobis Distance < 2√𝑑.
        if round == 4:
            while True:
                num_clusters = len(DS_dict) + len(CS_dict)
                pairs = list(product(list(DS_dict.keys()),list(CS_dict.keys())))
                for pair in pairs:
                    distance = cluster_Mahalanobis_Distance(DS_stat[pair[0]],CS_stat[pair[1]])
                    if distance < 2*sqrt(d):
                        DS_stat[pair[0]]['N'] += CS_stat[pair[1]]['N']
                        DS_stat[pair[0]]['SUM'] += CS_stat[pair[1]]['SUM']
                        DS_stat[pair[0]]['SUMSQ'] += CS_stat[pair[1]]['SUMSQ']
                        CS_stat.pop(pair[1])
                        DS_dict[pair[0]] += CS_dict[pair[1]]
                        CS_dict.pop(pair[1])
                        break
                if num_clusters == len(DS_dict) + len(CS_dict):    # No more combining
                    break

        # Output: “the number of the discard points”, “the number of the clusters in the compression set”,
        # “the number of the compression points”, and “the number of the points in the retained set”.
        num_DS = sum([len(value) for value in DS_dict.values()])
        num_cluster_CS = len(CS_dict)
        num_CS = sum([len(value) for value in CS_dict.values()])
        num_RS = len(RS)
        output += f"Round {round+1}: {num_DS},{num_cluster_CS},{num_CS},{num_RS}\n"

    # re-number cluster labels in CS
    CS_dict = {i+max(DS_dict)+1: CS_dict[k] for i, k in enumerate(sorted(CS_dict.keys()))}

    clustering_result = []
    for cluster in DS_dict:
        for point in DS_dict[cluster]:
            clustering_result.append([point, cluster])
    for cluster in CS_dict:
        for point in CS_dict[cluster]:
            clustering_result.append([point, -1])
    for outlier in RS:
        clustering_result.append([outlier, -1])

    clustering_result = sc.parallelize(clustering_result) \
                          .sortBy(lambda x: x[0]) \
                          .collect()

    output += "\nThe clustering results:"
    for point in clustering_result:
        output += "\n" + str(point[0]) + ',' + str(point[1])

    with open(output_file, 'w+') as file:
        file.write(output)
    file.close()
