from social_net import social_net
from utils import vertex_strength, local_clustering
from evo_utils import evolve
import pandas as pd
import networkx as nx

fin = evolve(generations=20, pop_size=5, elites=1, xover="random", mutation="uniform")

fin.to_csv("evo_test_run.csv")



# test = social_net(size=20, beliefs_size=20, tolerance=0.3, rationality=0.5, pressure=0.25, a=0.3)

# for i in range(200):
#     test.interaction_event()
#     if i%10 == 0: 
#         print(i)

# test.agent_connections()
# test.agent_coherence()

# G = nx.complete_graph(5)
# G.edges[1,0]['weight'] = 5
# G.edges[1,2]['weight'] = 1
# G.edges[1,3]['weight'] = 1
# G.edges[1,4]['weight'] = 1
# G.edges[2,3]['weight'] = 1
# G.edges[2,4]['weight'] = 1
# G.edges[3,4]['weight'] = 1
# print(vertex_strength(G, 0))
# print(local_clustering(G, 1))


