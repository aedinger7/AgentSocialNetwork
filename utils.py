import matplotlib.pyplot as plt
import networkx as nx
"""
Includes various functions for evaluating and manipulating network properties
"""
# Display network and clustering before and after pruning using networkx.draw_networkx and matplotlib
def compare_clustering(G):
    figure = plt.figure(figsize=(50,20))

    spring_pos = nx.spring_layout(G)

    gc= global_clustering(G)                           
    ax1 = plt.subplot(121)
    nx.draw_networkx(G, pos=spring_pos)
    ax1.set_title('Full Social Network: Clustering={:0.2f}'.format(gc), fontsize=32)

    gc= global_clustering(G, pruning_threshold=0.1)  
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
def global_clustering(G, pruning_threshold=0):
    if pruning_threshold>0:
        G=prune_graph(G, threshold=pruning_threshold)
    total = 0 
    for v in range(len(G)):
        total+=local_clustering(G,v)
    
    return total/len(G)


# Remove edges from G with weight below threshold weight of all edges in graph
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

# Automatically traverse network, flipping edges to create a completely coherent network
def coherify(G):
    for j in range(len(G)):
        for k in range(j+1,len(G)):
            for l in range(k+1,len(G)):
                if G.edges[j,k]['weight']*G.edges[k,l]['weight']*G.edges[j,l]['weight'] < 0:
                    G.edges[j,k]['weight'] *= -1
    return


