from random import randint
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, AffinityPropagation
import json
import os
from abstract_graph_edit_dist import compare
import matplotlib.pyplot as plt
import sys

class Cluster:
    def __init__(self, g =list()):
        self.list_of_graphs = g
        self.cluster_one_element()
        self.matrix_distance = self.create_distance_matrix()


    def get_centers_random(self):
        list_of_graph_center = list()
        i = 0
        sauv = list()
        while (i<self.number_of_center):
            nb_random = randint(0, len(self.list_of_graphs) -1)
            if nb_random not in sauv:
                g = self.list_of_graphs[nb_random]
                list_of_graph_center.insert(len(list_of_graph_center), g)
                i = i+ 1
                sauv.insert(len(sauv), nb_random)

        return list_of_graph_center

    def get_centers(self):
        list_graph_center = list()
        next = len(self.list_of_graphs)/self.number_of_center
        gr_l = self.sort(self.list_of_graphs)
        for i in range(0, self.number_of_center, 1):
            list_graph_center.append(gr_l[i * int(next)])

        return list_graph_center


    def sort(self, list_of_graph):
        data_list = list_of_graph
        new_list = list()
        while data_list:
            minimum = data_list[0]  # arbitrary number in list
            for x in data_list:
                if len(x) < len(minimum):
                    minimum = x
            new_list.append(minimum)
            data_list.remove(minimum)
        return new_list



    def distance_between_graphs(self, g1 ,g2):
        return self.matrix_distance[self.list_of_graphs.index(g1), self.list_of_graphs.index(g2)]

    def distance_between_graph_centers(self, g, list_of_centers):
        min = sys.maxsize
        graph = nx.DiGraph()
        for gr in list_of_centers:
            dist = self.distance_between_graphs(g, gr)
            if (dist ==0):
                return gr
            if (dist <min):
                min = dist
                graph = gr

        return graph

    def get_new_centers_simple(self, list_of_cluters):
        l = list()

        for r in list_of_cluters:
            if (len(r)>0):
                nb_random = randint(0, len(r))
                l.insert(len(l), r[nb_random-1])
        self.number_of_center = len(l)
        return l

    def get_graph_nearest_all(self, list_of_graphs):
        materix_distance = list()
        dist = 0
        if (len(list_of_graphs)>0):
            for e in range(0, len(list_of_graphs), 1):
                for f in range(0, len(list_of_graphs), 1):
                    dist = dist + compare(list_of_graphs[e], list_of_graphs[f])

                materix_distance.insert(e, dist)

                dist =0

            mini = min(materix_distance)
            index = materix_distance.index(mini)
            return list_of_graphs[index]
        else:
            return

    #get the graph the nearest to all graphs in the cluster
    def get_new_centers(self, list_of_cluters):
        l = list()
        for r in list_of_cluters:
            #calculated the graph nearest to all graphs
            gn = self.get_graph_nearest_all(r)
            l.insert(len(l), gn)

        return l

    #kmeans clustring
    def clustering(self):

        list_of_center_initial = self.get_centers_random()

        cpt =0
        number_of_iteration = 10
        list_cluters = list()

        while (cpt < number_of_iteration):
            print('cpt ', cpt)
            list_cluters = list()
            for e in list_of_center_initial:
                l = list()
                l.append(e)
                list_cluters.append(l)

            for g in self.list_of_graphs:
                if g not in list_of_center_initial:
                    gr= self.distance_between_graph_centers(g, list_of_center_initial)
                    print('gr', g.graph['name'], gr.graph['name'])
                    index = list_of_center_initial.index(gr)
                    list_cluters[index].append(g)
            for e in list_cluters:
                print('cluster ')
                for r in e:
                    print('gr ', r.graph['name'])
            list_of_center_initial = self.get_new_centers(list_cluters)
            cpt +=1

        self.print_cluster(list_cluters, 0)
        return list_cluters


    #kmedoid clustering
    def kmedoid_cluster(self):

        self.number_of_center = int(len(self.list_of_graphs) / 2)
        list_medoid = self.get_centers_random()
        list_cluters = list()

        for e in list_medoid:
            l = list()
            list_cluters.append(l)

        for g in self.list_of_graphs:
            if g not in list_medoid:
                gr = self.distance_between_graph_centers(g, list_medoid)
                index = list_medoid.index(gr)
                list_cluters[index].append(g)

        total_cost = self.total_cost(medoids=list_medoid, list_of_clusters=list_cluters)

        cpt = 0
        brk = False
        while(cpt < 200):

            for m in list_medoid:
                for o in self.list_of_graphs:
                    if o not in list_medoid:
                        self.swap(m, o, list_cluters, list_medoid)
                        if (self.total_cost(list_medoid, list_cluters)>total_cost):
                            self.undo_swap(m, o, list_cluters, list_medoid)
                        else:
                            total_cost = self.total_cost(medoids=list_medoid, list_of_clusters=list_cluters)
                            brk = True
                            break
                if brk == True:
                    brk = False
                    break
            cpt += 1

        data = self.write_configuration(list_medoid, list_cluters, total_cost)
        list_cluters, list_medoid = self.get_from_file(data)

        for e in list_medoid:
            list_cluters[list_medoid.index(e)].append(e)

        self.print_cluster(list_cluters, 1)

        return list_cluters, list_medoid


    #not used because no effect
    def reaffect_clusters_with_no_element(self, list_cluters, list_medoid):

            for c in list_cluters:
                if len(c) == 0:
                    #no element in this clusters
                    m = list_medoid.pop(list_cluters.index(c))
                    list_cluters.pop(list_cluters.index(c))
                    if m not in list_medoid:
                        gr = self.distance_between_graph_centers(m, list_medoid)
                        index = list_medoid.index(gr)
                        list_cluters[index].append(m)
            return list_cluters, list_medoid


    def get_from_file(self, data):
        list_medoid = list()
        list_cluters = list()
        for e in self.list_of_graphs:

            if e.graph['name'] in data['config'][0]['medoid']:
                list_medoid.append(e)
        i = 0
        for e in data['config'][0]['clusters']:
            list_cluters.append(list())
            for g in self.list_of_graphs:
                if g.graph['name'] in e:
                    list_cluters[i].append(g)

            i += 1

        return list_cluters, list_medoid

    #swap medoid and non medoid in clusters and centers
    def swap(self, medoid, non_medoid, list_of_clusters, list_of_medoids):

        list_of_medoids.remove(medoid)
        list_of_medoids.append(non_medoid)

        for c in list_of_clusters:
            if non_medoid in c:
                c.remove(non_medoid)
                c.append(medoid)
                list_of_clusters[list_of_clusters.index(c)] = c
                return
        return


    #rollback the swap
    def undo_swap(self, medoid, non_medoid, list_of_clusters, list_of_medoids):

        list_of_medoids.remove(non_medoid)
        list_of_medoids.append(medoid)

        for c in list_of_clusters:
            if medoid in c:
                c.remove(medoid)
                c.append(non_medoid)
                list_of_clusters[list_of_clusters.index(c)] = c
                return
        return

    #not a good function
    def is_stop_time(self, list_pervious_costs):

        if len(list_pervious_costs) ==0:
            return False
        for e in list_pervious_costs:
            for f in list_pervious_costs:
                if e != f:
                    element = list_pervious_costs.pop()
                    list_pervious_costs = list()
                    list_pervious_costs.append(element)
                    return False

        return True

    def cost(self, medoid, list_non_medoid):
        sum = 0
        for n in list_non_medoid:
            sum = sum + self.matrix_distance[self.list_of_graphs.index(medoid), self.list_of_graphs.index(n)]

        return sum

    def total_cost(self, medoids, list_of_clusters):
        sum = 0
        i = 0
        for m in medoids:
            sum = sum + self.cost(m, list_of_clusters[i])
            i += 1

        return sum


    def cluster_one_element(self):
        cluster = list()
        for e in self.list_of_graphs:
            if len(e.edges()) == 0:
                cluster.append(e)
        for e in cluster:
            self.list_of_graphs.remove(e)
        list_graph = list()
        list_graph.append(cluster)
        self.print_cluster(list_graph, 10)
        return


    def compute_distance_between_centers(self, list_of_centers):
        distance_matrix = list()
        for e in list_of_centers:
            som = 0
            for f in list_of_centers:
                som = som + compare(e, f)
            distance_matrix.append(som)

        return
    def create_distance_matrix(self):

        matrix = np.zeros((len(self.list_of_graphs), len(self.list_of_graphs)))
        i = 0
        for e in self.list_of_graphs:
            j = 0
            for g in self.list_of_graphs:
                matrix[i, j] = compare(e, g)
                j += 1
            i += 1
        return matrix

    def dbscan_clustering(self):
        db = DBSCAN(eps=0.2, min_samples=3, metric='precomputed').fit(self.matrix_distance)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(n_clusters)
        # print(labels)
        clusters = list()
        graphs = {}
        print("noise ")
        for item in range(len(labels)):
            if labels[item] == -1:
                print(self.list_of_graphs[item])
        n = 0
        for item in labels:
            if item != -1:
                if item in clusters:
                    graphs[item].append(self.list_of_graphs[n])
                else:
                    clusters.append(item)
                    graphs[item] = [self.list_of_graphs[n]]
            n += 1
        list_graph = list()
        for c in clusters:
            element = list()
            for e in graphs[c]:
                element.append(e)
            list_graph.append(element)
        self.print_cluster(list_graph, 2)

    # the best results at that time
    def affinity_propagation(self):
        af = AffinityPropagation(affinity='precomputed').fit(self.matrix_distance)
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
                clusters[item].append(self.list_of_graphs[n])
                graphs[item].append(self.list_of_graphs[n])
            else:
                clusters[item] = [self.list_of_graphs[n]]
                graphs[item] = [self.list_of_graphs[n]]
            n += 1
        # test with centers propsed by the moethod
        graph_centrers = list()
        for i in cluster_centers_indices:
            graph_centrers.append(self.list_of_graphs[i])

        list_graph = list()
        for item in clusters:
            print("Cluster ", item, len(clusters[item]))
            element = list()
            for i in clusters[item]:
                element.append(i)
            list_graph.append(element)

        #self.print_graph_centers(graph_centrers)
        self.print_cluster(list_graph, 3)
        return clusters, graphs

    def print_graph_centers(self, graph_centers):
        plt.figure()
        plt.axis('off')
        i = 0
        for e in graph_centers:
            pos = nx.circular_layout(e)
            for k, v in pos.items():
                v[0] = v[0] + 5 * i
            colors = e.nodes(data='color')
            values = [colors[node] for node in e.nodes()]
            nx.draw_networkx_nodes(e, pos, nodelist=e.nodes(), node_color=values, cmap=plt.get_cmap('jet'))
            nx.draw_networkx_edges(e, pos, edgelist=e.edges())
            nx.draw_networkx_edge_labels(e, pos, edge_labels=nx.get_edge_attributes(e, 'relationship'))
            nx.draw_networkx_labels(e, pos, labels=nx.get_node_attributes(e, 'label'))
            plt.title('list of centers')
            i += 1

        return


    def print_cluster(self, list_clusters, number):

        if number == 0:
            path = 'kmeans'
        else:
            if number == 1:
                path = 'kmedoids'
            else:
                if number == 2:
                    path = 'dbscan'
                else:
                    if number == 3:
                        path = 'affinity'
                    else:
                        path = 'noedge'
        if not os._exists(path):
            os.makedirs(path)

        for cl in list_clusters:
            newpath = path+'/cluster'+str(list_clusters.index(cl))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            for elem in cl:
                    plt.figure()
                    plt.axis('off')
                    pos = nx.circular_layout(elem)
                    colors = elem.nodes(data='color')
                    values = [colors[node] for node in elem.nodes()]
                    nx.draw_networkx_nodes(elem, pos, nodelist=elem.nodes(), node_color=values, cmap=plt.get_cmap('jet'))
                    nx.draw_networkx_edges(elem, pos, edgelist=elem.edges())
                    nx.draw_networkx_edge_labels(elem, pos, edge_labels=nx.get_edge_attributes(elem, 'relationship'))
                    nx.draw_networkx_labels(elem, pos, labels=nx.get_node_attributes(elem, 'label'))
                    plt.title(str(elem.graph['name']))
                    plt.savefig(newpath+'/'+str(elem.graph['name']))

        return


    def write_configuration(self, medmoids, clusters, cost):
        """
        #this function write the best configuration in json file
        configuration means the centers and clusters
        :return: None
        """
        m = list()
        for r in medmoids:
            m.append(r.graph['name'])

        c = list()
        for cl in clusters:
            ll = list()
            for i in cl:
                ll.append(i.graph['name'])
            c.append(ll)

        data = {}
        data['config'] = []
        data['config'].append({
            'cost': cost,
            'medoid': m,
            'clusters': c
        })
        write =False
        print(data)
        with open('./config.json', 'r') as outfile:
            config = json.load(outfile)
            if float(config['config'][0]['cost']) > cost and len(m) == len(config['config'][0]['medoid']):
                write = True

        if (write == True):
            self.write_in_file(data)
            return data
        else:
            return config

    def write_in_file(self, data):
        # a good configuration must be stored
        with open('./config.json', 'w') as out:
            json.dump(data, out)
            print('configuration saved')

        return