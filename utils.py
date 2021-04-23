"""
Includes various functions for evaluating and manipulating network properties
"""

# Vertex strength metric from Barrat et al 2004
def vertex_strength(G, node):
    s = 0
    for i in range(len(G)):
        if i != node and (G.edges[node,i]):
            s+=G.edges[node,i]['weight']
        
    return s


# Local clustering coefficient from Barrat et al 2004
def local_clustering(G, node):
    n = 1/(vertex_strength(G,node)*(G.degree[0]-1))
    s = 0
    for j in range(len(G)):
        for h in range(len(G)):
            if j!=h and j!=node and h!=node:
                if (G.edges[j,h]) and (G.edges[node,j]) and (G.edges[node,h]):   
                    s+=(G.edges[node,j]['weight'] + G.edges[node,h]['weight'])/2
                
    return n*s


# Global clustering coefficient from Barrat et al 2004
def global_clustering(G):
    total = 0 
    for v in range(len(G)):
        total+=local_clustering(G,v)
    
    return total/len(G)


# Remove edges with weight below mean weight of all edges in graph
def prune_graph(G):
    weights = [G.edges()[edge]['weight'] for edge in G.edges]
    mean_weight = sum(weights)/len(weights)

    print(len(G.edges))

    for (u, v) in G.edges():
            if G.edges[u,v]['weight'] < mean_weight:
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


