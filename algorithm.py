
from sklearn.cluster import KMeans, AffinityPropagation, estimate_bandwidth, MeanShift, DBSCAN, spectral_clustering
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# filter nodes in graphs, keep only Activity nodes
def filter_activity(g):
        gr = g.copy()
        tyes = g.nodes(data='type', default=None)
        for e in g.nodes():
            if tyes[e] != 'is_activity':
                gr.remove_node(e)
        #print('activity number of nodes', len(gr.nodes()), 'number of edges ', len(gr.edges()))
        return gr

# Kmeans clustering
# params: number_of_cluster
def kmeans(data_set, number_of_cluster = 5):
    kmeans = KMeans(n_clusters = number_of_cluster)
    print('infos ', kmeans)
    y = kmeans.fit_predict(data_set)
    print('labels ', y)
    # to print the element in clusters
    clusters = {}
    graphs = {}
    n = 0
    for item in y:
        if item in clusters:
            clusters[item].append(data_set[n])

        else:
            clusters[item] = [data_set[n]]
        n += 1

    print_graphs(clusters)

# print clusters and the graphs in theirs clusters
def print_graphs(clusters):
    print(clusters)
    for item in clusters:
        print("Cluster ", item, len(clusters[item]))
        for i in clusters[item]:
            print(i)

# print graphs and their commun part
def printf(cluster, number):
    plt.figure()
    plt.axis('off')
    i = 0
    for e in cluster:
            pos = nx.circular_layout(e)

            for k, v in pos.items():
                # Shift the x values of every node by 10 to the right
                v[0] = v[0] + 5 * i
            colors = e.nodes(data='color')
            values = [colors[node] for node in e.nodes()]
            nx.draw_networkx_nodes(e, pos, nodelist=e.nodes(), node_color=values, cmap=plt.get_cmap('jet'))
            nx.draw_networkx_edges(e, pos, edgelist=e.edges())
            nx.draw_networkx_edge_labels(e, pos, edge_labels=nx.get_edge_attributes(e, 'relationship'))
            nx.draw_networkx_labels(e, pos, labels=nx.get_node_attributes(e, 'label'))
            plt.title('cluster '+ str(number))
            i += 1

    plt.figure()
    plt.axis('off')

# affinity propagation clustering
# params: preference default value -50 not specified the median
# the best results at that time
def affinity_propagation(data_set, preference = -50):
    af = AffinityPropagation(preference=preference).fit(data_set)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('labels ', labels)
    print('infos af', af)
    # to print the element in clusters
    clusters = {}
    graphs = {}
    n = 0
    for item in labels:
        if item in clusters:
            clusters[item].append(data_set[n])

        else:
            clusters[item] = [data_set[n]]

        n += 1

    for item in clusters:
        print("Cluster ", item, len(clusters[item]))
        for i in clusters[item]:
            print(i)
    return clusters, graphs


#spectral clustering
def spectral_clustering_graph(data_set):

    labels = spectral_clustering(data_set)
    print(labels)

#no meaning
def get_fragment_from_activity(graph):

    for e in graph.nodes():
        print(str(e))
        return