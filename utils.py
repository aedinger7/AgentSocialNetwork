# pylint: disable=unused-variable
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
import networkx.algorithms.community as nx_comm

"""
Includes various functions for evaluating and manipulating network properties
"""

# Display network and clustering coefficients before and after pruning using networkx.draw_networkx and matplotlib
# Used to evaluate effect of pruning on clustering coefficient and network structure
def compare_clustering(G, pruning_threshold=0.1):
    figure = plt.figure(figsize=(50,20))

    spring_pos = nx.spring_layout(G)

    gc= 1-global_clustering(G)                           
    ax1 = plt.subplot(121)
    nx.draw_networkx(G, pos=spring_pos)
    ax1.set_title('Full Social Network: Clustering={:0.2f}'.format(gc), fontsize=32)  

    gc= 1-global_clustering(G, pruning_threshold=pruning_threshold)  
    ax2 = plt.subplot(122)
    nx.draw_networkx(prune_graph(G, type='threshold', threshold=.1), pos=spring_pos)
    ax2.set_title('Pruned Social Network: Clustering={:0.2f}'.format(gc), fontsize=32)


# Vertex strength metric from Barrat et al 2004
def vertex_strength(G, node):
    s = 0
    for (u,v) in G.edges(node):
        if(G.edges[u,v]):
            s+=G.edges[u,v]['weight']
        
    return s


# Local clustering coefficient from Barrat et al 2004
# Calculates clustering around node in graph G
def local_clustering(G, node):
    deg = 0
    for (u,v) in G.edges(node):
        if G.edges[u,v] != 0:
            deg += 1

    s = vertex_strength(G,node)
    if s==0 or deg-1==0:
        return 1

    n = 1/(s*(deg-1))
    weights = 0
    for (u,v) in G.edges(node):
        for (x,y) in G.edges(node):
            if (u,v)!=(x,y) and G.has_edge(v, y):
                if G.edges[u,v]!=0 and G.edges[x,y]!=0 and G.edges[v,y]!=0:
                    weights+=(G.edges[u,v]['weight'] + G.edges[x,y]['weight'])/2
                
    return n*weights


# Global clustering coefficient from Barrat et al 2004
# Average clustering over all nodes in graph G
# pruning_threshold - int: if >0, removes all edges under threshold before calculating clustering
def global_clustering(G, pruning_threshold=0):
    if pruning_threshold>0:
        G=prune_graph(G, threshold=pruning_threshold)
    total = 0 
    for v in range(len(G)):
        total+=local_clustering(G,v)
    
    return total/len(G)


# Community detection algorithm implementation from Clauset et al 2004
# pruning_threshold - int: removes all edges under threshold before calculating clustering
def modularity(G, pruning_threshold=0):
    if pruning_threshold>0:
        G=prune_graph(G, threshold=pruning_threshold)
    partition = community_louvain.best_partition(prune_graph(G, threshold=.1))
    partition_sets = [{x for x in partition.keys() if partition[x] == i} for i in set(partition.values())]
    return nx_comm.modularity(G, partition_sets)


# Remove edges from G with weight below threshold weight of all edges in graph
# type - str: 'threshold' to specify value, 'mean' to remove nodes beneath mean edge weight of graph
# inplace - boolean: True directly modifies the given graph object, False makes and returns a copy of the graph
def prune_graph(G, type='threshold', threshold=0.1, inplace=False):
    if type=='mean':
        weights = [G.edges()[edge]['weight'] for edge in G.edges]
        threshold = sum(weights)/len(weights)
    
    if not inplace:
        G=G.copy()

    for (u, v) in G.edges():
        if G.edges[u,v]['weight'] < threshold:
            G.remove_edge(u,v)
                
    return(G)


# Network coherence measure from Rodriguez et al
# Counts number of coherent - noncoherent triples, doesn't count incomplete triples
# Weighted - boolean: If true, takes edge weights into account. If False, only considers negative or positive values
def network_coherence(G, weighted=False):
    e = 0
    count = 0
    for j in range(len(G)):
        for k in range(j+1,len(G)):
            for l in range(k+1,len(G)):
                if weighted:
                    e += G.edges[j,k]['weight']*G.edges[k,l]['weight']*G.edges[j,l]['weight']
                else:
                    test = G.edges[j,k]['weight']*G.edges[k,l]['weight']*G.edges[j,l]['weight']
                    if test < 0:
                        e += -1
                        count += 1
                    if test > 0:
                        e += 1
                        count+=1
    if count != 0:
        return e/count
    else:
        return False

# Automatically traverse network G, flipping edges to create a completely coherent network
def coherify(G):
    for j in range(len(G)):
        for k in range(j+1,len(G)):
            for l in range(k+1,len(G)):
                if G.edges[j,k]['weight']*G.edges[k,l]['weight']*G.edges[j,l]['weight'] < 0:
                    G.edges[j,k]['weight'] *= -1
    return


