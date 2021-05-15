import logging
import sys
from operator import itemgetter
import networkx as nx
from networkx.algorithms import degree_centrality
import pandas as pd
import numpy as np

# Dataset file names:
# Wiki-Vote
# CA-GrQc
# p2p-Gnutella08
# facebook_combined


filename = input("Input the dataset Filename: ")
dataset_file_common_path = ('dataset\\' + filename + '.txt')

col_names = ['FromNode', 'ToNode']
try:
    if filename == 'facebook_combined':
        df = pd.read_csv(dataset_file_common_path, delimiter=r"\s+", names=col_names)
    else:
        df = pd.read_csv(dataset_file_common_path, sep="\t", skiprows=[0, 1, 2, 3], names=col_names)
except IOError:
    print('\nError....the dataset filename entered is incorrect or not found...please check and try again.')
    sys.exit()

# save the results in an output file
resultFile = open('results\\' + filename + '_Result_Output.txt', "w")
resultFile.write('# Directed graph (each unordered pair of nodes is saved once): {}'.format(filename + '.txt') + '\n\n')

# function to get unique values
# def unique(list1):
#    x = np.array(list1)
#    return np.unique(x)


nodes = df.to_numpy().flatten().tolist()
node_names = np.unique(np.array(nodes)).tolist()  # Get a list of only the node names (unique node values)
edges = [tuple(e) for e in df.values]

# print('Nodes:', len(node_names))
# print('Edges:', len(edges))
# resultFile.write('Nodes: {} '.format(len(node_names)) + '\n')
# print('\n')

# This will create a new Graph object
G = nx.Graph()

# add your lists of nodes and edges like so:
G.add_nodes_from(node_names)
G.add_edges_from(edges)

# get basic information about your newly-created network using the info function:
print(nx.info(G), '\n')
resultFile.write('# Information about the newly-created network is:\n{}'.format(nx.info(G)) + '\n\n')

# Compute the following node measures:
print('# Compute the following node measures: \n')
resultFile.write('# Compute the following node measures:' + '\n\n')

# a. Degree Centrality (normalized)
degree_centrality = degree_centrality(G)
sorted_degree = sorted(degree_centrality.items(), key=itemgetter(1), reverse=True)
print("# Top 10 nodes by degree centrality:")
resultFile.write('# Top 10 nodes by degree centrality:' + '\n')

for d in sorted_degree[:10]:
    print(d)
    resultFile.write('{} '.format(d) + '\n')
print('\n')
resultFile.write('{} '.format('') + '\n')

# b. Closeness Centrality (normalized)
# closeness_centrality = closeness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
sorted_closeness = sorted(closeness_centrality.items(), key=itemgetter(1), reverse=True)

print("# Top 10 nodes by closeness centrality:")
resultFile.write('# Top 10 nodes by closeness centrality: ' + '\n')
for b in sorted_closeness[:10]:
    print(b)
    resultFile.write('{} '.format(b) + '\n')
print('\n')
resultFile.write('{} '.format('') + '\n')

# c. Betweenness Centrality
# betweenness_centrality = betweenness_centrality(G[, k, normalized, …])
betweenness_dict = nx.betweenness_centrality(G)  # Run betweenness centrality

# Assign each to an attribute in your network
nx.set_node_attributes(G, betweenness_dict, 'betweenness')
sorted_betweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)

print("# Top 10 nodes by betweenness centrality:")
resultFile.write('# Top 10 nodes by betweenness centrality: ' + '\n')
for b in sorted_betweenness[:10]:
    print(b)
    resultFile.write('{} '.format(b) + '\n')
print('\n')
resultFile.write('{} '.format('') + '\n')

# d. Eigenvector Centrality
eigenvector_dict = nx.eigenvector_centrality(G)  # Run eigenvector centrality
nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
sorted_eigenvector = sorted(eigenvector_dict.items(), key=itemgetter(1), reverse=True)

print("# Top 10 nodes by Eigenvector centrality:")
resultFile.write('# Top 10 nodes by Eigenvector centrality:' + '\n')
for i in sorted_eigenvector[:10]:
    print(i)
    resultFile.write('{}'.format(i) + '\n')
print('\n')
resultFile.write('{} '.format('') + '\n')

# e. Pagerank Centrality (with d=0.85)
pagerank_dict = nx.pagerank(G, alpha=0.85)  # Run eigenvector centrality
nx.set_node_attributes(G, pagerank_dict, 'eigenvector')
sorted_pagerank = sorted(pagerank_dict.items(), key=itemgetter(1), reverse=True)

print("# Top 10 nodes by Pagerank centrality:")
resultFile.write('# Top 10 nodes by Pagerank centrality: ' + '\n')
for i in sorted_pagerank[:10]:
    print(i)
    resultFile.write('{} '.format(i) + '\n')
print('\n')
resultFile.write('{} '.format('') + '\n')

# f. Clustering Coefficient
clustering_dict = nx.clustering(G)
nx.set_node_attributes(G, clustering_dict, 'eigenvector')
sorted_clustering = sorted(clustering_dict.items(), key=itemgetter(1), reverse=True)

print("# Top 500 nodes by Clustering centrality:")
resultFile.write('# Top 500 nodes by Clustering centrality' + '\n')
for x in sorted_clustering[:1500]:
    print(x)
    resultFile.write('{} '.format(x) + '\n')
print('\n')
resultFile.write('{} '.format('') + '\n')
