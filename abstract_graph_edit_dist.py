# -*- coding: UTF-8 -*-
from __future__ import print_function
from scipy.optimize import linear_sum_assignment
import sys
import numpy as np
import networkx as nx


def compare(g1, g2, print_details=False):

    ged = AbstractGraphEditDistance(g1, g2)
    if print_details:
        ged.print_matrix()
    print('distance =', ged.normalized_distance())

    return ged.normalized_distance()

# get the commun part between two graphs
def get_commun_part_graph(g1, g2):
        ged = AbstractGraphEditDistance(g1, g2)
        matrix = np.zeros((len(g1.edges()), len(g2.edges())))
        i = 0
        for e1 in g1.edges(data=True):
                j = 0
                for e2 in g2.edges(data=True):
                    a = ged.deep(g1, e1[0])
                    b = ged.deep(g2, e2[0])
                    matrix[i, j] = max(a, b) - min(a, b)
                    j += 1
                i += 1
        types1 = g1.nodes(data='type', default=None)

        name = g1.graph['name']+g2.graph['name']

        g = nx.DiGraph(name=name)
        colors = g1.nodes(data='color')
        row_ind, col_ind = linear_sum_assignment(matrix)
        edges = list(g1.edges(data=True))
        for i in range(len(row_ind)):
            if (matrix[row_ind[i], col_ind[i]] == 0):
                g.add_node(edges[row_ind[i]][0],
                           label=g1.nodes[edges[row_ind[i]][0]]['label'],
                           type=types1[edges[row_ind[i]][0]],
                           color=colors[edges[row_ind[i]][0]])
                g.add_node(edges[row_ind[i]][1],
                           label=g1.nodes[edges[row_ind[i]][1]]['label'],
                           type=types1[edges[row_ind[i]][1]],
                           color=colors[edges[row_ind[i]][1]])
                g.add_edge(edges[row_ind[i]][0], edges[row_ind[i]][1],
                           relationship=edges[row_ind[i]][2]['relationship'])
        return g



class AbstractGraphEditDistance(object):
    def __init__(self, g1, g2):
        if (len(g1)<len(g2)):
            self.g1 = g1
            self.g2 = g2
        else:
            self.g2 = g1
            self.g1 = g2
        self.g = nx.DiGraph()#the graph containing the commun part between the two graphs :p


    def normalized_distance(self):
        """
        Returns the graph edit distance between graph g1 & g2
        The distance is normalized on the size of the two graphs.
        This is done to avoid favorisation towards smaller graphs
        """
        avg_graphlen = (len(self.g1) + len(self.g2)) / 2
        return self.distance() / avg_graphlen

    def distance(self):
        return sum(self.edit_costs())

    def edit_costs(self):
        cost_matrix = self.create_cost_matrix()
        row_ind,col_ind = linear_sum_assignment(cost_matrix)
        return [cost_matrix[row_ind[i]][col_ind[i]] for i in range(len(row_ind))]

    def create_cost_matrix(self):
        """
        Creates a |N+M| X |N+M| cost matrix between all nodes in
        graphs g1 and g2
        Each cost represents the cost of substituting,
        deleting or inserting a node
        The cost matrix consists of four regions:

        substitute 	| insert costs
        -------------------------------
        delete 		| delete -> delete

        The delete -> delete region is filled with zeros
        """

        n = len(self.g1)
        m = len(self.g2)

        cost_matrix = np.zeros((n+m,n+m))

        nodes1 = list(self.g1.nodes())
        nodes2 = list(self.g2.nodes())

        for i in range(n):
            for j in range(m):
                cost_matrix[i,j] = self.substitute_cost(nodes1[i], nodes2[j])

        for i in range(m):
            for j in range(m):
                cost_matrix[i+n,j] = self.insert_cost(nodes2[i], nodes2[j])

        for i in range(n):
            for j in range(n):
                cost_matrix[j,i+m] = self.delete_cost(nodes1[i], nodes1[j])

        self.cost_matrix = cost_matrix
        return cost_matrix
    #compute the cost of changing label + difference between edges of nodes
    def substitute_cost(self, node1, node2):
        return self.relabel_cost(node1, node2) + self.edge_diff(node1, node2)

    #compute the cost of changing label of node1 from the first graph by node2 from the second graph
    def relabel_cost(self, node1, node2):

        label1 = self.g1.node[node1].get('label')
        label2 = self.g2.node[node2].get('label')
        if label1 == label2:
            #distiguish between label
            #get type if class
            if label1 == "['Class']":
                types1 = self.g1.nodes(data='type', default=None)
                types2 = self.g2.nodes(data='type', default=None)
                type1 = types1[node1]
                type2 = types2[node2]
                if type1 == type2:
                    #class with same type
                    return 0
                else:
                    return 20
            return 0
        else:
            return 20

    #insert an edge in the second graph
    def insert_cost(self, node1, node2):
        label1 = self.g2.nodes[node1]['label']
        label2 = self.g2.nodes[node2]['label']
        if label1 == label2:
            #the same node: no insert
            return 0
        else:
            #insert a new edge Extend or Uses or Implements
            #verify if the edge exist in the graph at first
            try :
                relationship = self.g1.edges[node1, node2]['relationship']
                #the edge exist in the graph ->
                return 0
            except KeyError:
                try:
                    relationship = self.g1.edges[node2, node1]['relationship']
                    # the edge exist in the graph <-
                    return 0
                except KeyError:
                #this edge doesn't exist in the graph "[]"
                    if (label1 == "['Class']" and label2 == "['Class']") :
                        #add an edge extend or impelements or uses (the only relationship between classes
                        #this edge is the most important in the graph
                        types2 = self.g2.nodes(data='type', default=None)
                        type1 = types2[node1]
                        type2 = types2[node2]
                        if type1 != None and type2 != None:
                            #we need to avoid this type de operation (inserting an between two class fundamental in Android
                            return 20
                        return 15
                    if (label1 == "['Method']" and label2 == "['Class']") or (label1 == "['Class']" and label2 == "['Method']"):
                        #add an edge Class_Owns_Method
                        return 10
                    if (label1 == "['Class']" and label2 == "['Variable']") or (label2 == "['Class']" and label1 == "['Variable']"):
                        # add an edge Class_Owns_Variable
                        return 5
                    if (label1 == "['Method']" and label2 == "['Variable']") or (label2 == "['Method']" and label1 == "['Variable']"):
                        # add an edge Method_Owns_Variable
                        return 5
            return 20 #not acceptable

    #delete an edge from the first graph
    def delete_cost(self, node1, node2):
        label1 = self.g1.nodes[node1]['label']
        label2 = self.g1.nodes[node2]['label']

        if label1 == label2:
            #delete the node from the graph
            #must be modified
            if label1=="['App']":#impossible to remove the node root
                return 20
            else:
                if label1=="['Class']":
                    return 15
                else:
                    if label1 =="['Method']":
                        return 10
                    else:
                        if label1 == "['Variable']":
                            return 5
                        else:
                            return 0 #Arguments and messages are not importants
        else:
            #removing an edge from the second graph
            #using matrix_adjency_iterator

            try :
                relationship = self.g1.edges[node1, node2]['relationship']
            except KeyError:
                try:
                    relationship = self.g1.edges[node2, node1]['relationship']
                except KeyError:
                    relationship = '' #this edge doesn't exist in the graph
            #the edge exist in the graph
            if (relationship == 'APP_OWNS_CLASS') \
                    or (relationship == 'EXTENDS') \
                    or (relationship == 'IMPLEMENTS') \
                    or (relationship == 'USES' and (label1 == "['Class']" or label2 == "['Class']")):
                #APP->CLASS or CLASS->CLASS
                types1 = self.g1.nodes(data='type', default=None)
                type1 = types1[node1]
                type2 = types1[node2]
                if type1 != None and type2 != None:
                    # we need to avoid this type de operation (inserting an between two class fundamental in Android
                    return 20
                return 15  #severe
            if (relationship == 'CLASS_OWNS_METHOD'):
                 #CLASS->METHOD
                return 15
            else:
                if (relationship == 'CLASS_OWNS_VARIABLE') or (relationship == 'METHOD_OWNS_VARIABLE'):
                    #CLASS->VARIABLE or METHOD->VARIABLE
                    return 10
                else:
                    if (relationship == 'USES'):
                        return 5
                    else:
                        return 0

    #the core method compute the diference between two nodes and theirs edges
    def edge_diff(self, node1, node2):

        edges1 = list(self.g1.successors(node1))
        edges2 = list(self.g2.successors(node2))

        types1 = self.g1.nodes(data='type', default=None)
        types2 = self.g2.nodes(data='type', default=None)

        type1 = types1[node1]
        type2 = types2[node2]

        if (self.deep(self.g1, node1) == 2) and (self.deep(self.g2, node2) == 2):
            #compare between all edges of the two graphs
            cost = list()
            for e1 in edges1:
                label1 = self.g1.nodes[e1]['label']
                relationship1 = self.g1.edges[node1, e1]['relationship']
                label2 = self.g1.nodes[node1]['label']
                m = []
                type11 = types1[e1]
                for e2 in edges2:
                    label12 = self.g2.nodes[e2]['label']
                    relationship2 = self.g2.edges[node2, e2]['relationship']
                    label22 = self.g2.nodes[node2]['label']
                    if (label2 == label22 and label1 == label12 and relationship1 == relationship2):
                        type22 = types2[e2]
                        if (label1 == "['Class']") and (type1 != type2 or type22 != type11):
                            m.append(1)
                        else:
                            m.append(0)
                    else:
                        m.append(10)
                cost.append(sum(m))
            return cost.index(min(cost))
        else:
            if (len(edges1) == 0) or (len(edges2) == 0):
                return 20
            else:
                return 15

        cost = list()
        indexes = list()
        for e1 in edges1:
            matrix = list()
            for e2 in edges2:
                matrix.append(self.edge_diff(e1, e2) + self.difference_edge(node1, e1, node2, e2))
                indexes.append([e1, e2])
            cost.append(sum(matrix))
        return cost.index(min(cost))

    #get the deep of a graph g taking initial_node as starting node
    def deep(self, g, initial_node):

        edges = list(g.successors(initial_node))
        return len(edges)
        if (len(edges) == 0):
            return 1
        else:
            m = []
            for e in edges:
                m.append(self.deep(g,e))

            return max(m) + 1

    #compute the difference between two edges (node1->e1) and (node2->e2)
    def difference_edge(self, node1, e1, node2, e2):
        label1 = self.g1.nodes[node1]['label']
        label2 = self.g1.nodes[e1]['label']
        relationship1 = self.g1.edges[node1, e1]['relationship']

        label11 = self.g2.nodes[node2]['label']
        label22 = self.g2.nodes[e2]['label']
        relationship2 = self.g2.edges[node2, e2]['relationship']

        types1 = self.g1.nodes(data='type', default=None)
        types2 = self.g2.nodes(data='type', default=None)
        t1 = types1[node1]
        t11 = types1[e1]
        t2 = types2[node2]
        t22 = types2[e2]
        if (label1 == label11 and label2 == label22 and relationship1 == relationship2):
            if (label1 == "['Class']") and (t1 != t2 or t22 != t11):
                return 20
            return 0
        else:
            return 20

    # get the commun part between the two graphs
    def get_commun_part_graph(self):

        matrix = np.zeros((len(self.g1.edges()), len(self.g2.edges())))
        types1 = self.g1.nodes(data='type', default=None)
        types2 = self.g2.nodes(data='type', default=None)
        i = 0
        for e1 in self.g1.edges(data = True):
            j = 0

            label1 = self.g1.nodes[e1[0]]['label']
            label2 = self.g1.nodes[e1[1]]['label']
            relationship1 = self.g1.edges[e1[0], e1[1]]['relationship']

            for e2 in self.g2.edges(data = True):

                label11 = self.g2.nodes[e2[0]]['label']
                label22 = self.g2.nodes[e2[1]]['label']
                relationship2 = self.g2.edges[e2[0], e2[1]]['relationship']

                if (label1 == label11 and label2 == label22 and relationship1 == relationship2):
                    type1 = types1[e1[0]]
                    type2 = types2[e2[0]]
                    type11 = types1[e1[1]]
                    type22 = types2[e2[1]]

                    if (type1 != type2  and label1 == "['Class']") or (label2 == "['Class']" and type11 != type22):
                        matrix[i, j] = 1


                    else:
                        matrix[i, j] = 0

                else:
                    matrix[i, j] = 1

                j += 1
            i += 1
        colors = self.g1.nodes(data='color')
        row_ind, col_ind = linear_sum_assignment(matrix)
        edges = list(self.g1.edges(data = True))
        for i in range(len(row_ind)):
            if (matrix[row_ind[i], col_ind[i]] == 0):
                self.g.add_node(edges[row_ind[i]][0],
                                label = self.g1.nodes[edges[row_ind[i]][0]]['label'],
                                type = types1[edges[row_ind[i]][0]],
                                color = colors[edges[row_ind[i]][0]])
                self.g.add_node(edges[row_ind[i]][1],
                                label = self.g1.nodes[edges[row_ind[i]][1]]['label'],
                                type = types1[edges[row_ind[i]][1]],
                                color = colors[edges[row_ind[i]][1]])
                self.g.add_edge(edges[row_ind[i]][0], edges[row_ind[i]][1],
                                relationship = edges[row_ind[i]][2]['relationship'])

    def print_matrix(self):
        print("cost matrix:")
        for column in self.create_cost_matrix():
            for row in column:
                if row == sys.maxint:
                    print ("inf\t")
                else:
                    print ("%.2f\t" % float(row))
            print("")


