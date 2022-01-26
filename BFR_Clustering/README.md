In this project I implement the Bradley-Fayyad-Reina (BFR) algorithm to cluster data.


Since the BFR algorithm has a strong assumption that the clusters are normally distributed with independent dimensions, I use synthetic datasets with random centroids, some data points with the centroids, and some standard deviations that form the clusters. 
There are also outliers. Data points which are outliers belong to clusters that is named or indexed as “-1”.


In BFR, there are three sets of points that you need to keep track of:
 
Discard set (DS), Compression set (CS), Retained set (RS)


For each cluster in the DS and CS, the cluster is summarized by: 
 
N: The number of points

SUM: the sum of the coordinates of the points 

SUMSQ: the sum of squares of coordinates


The output file is a text file, containing the following information.

a. The intermediate results. Each line is started with “Round {𝑖}:” and 𝑖 is the count for the round. Then the values for “the number of the discard points”, “the number of the clusters in the compression set”, “the number of the compression points”, and “the number of the points in the retained set”.

b. The clustering results, including the data points index and their clustering results after the BFR algorithm. The cluster of outliers should be represented as -1.
