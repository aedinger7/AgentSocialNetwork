from social_net import social_net
from utils import vertex_strength, local_clustering, compare_clustering, global_clustering, prune_graph, modularity
from evo_utils import evolve
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
"""Example run and evaluation usage for evolving a social_net object
"""

# Evolve an agent 
fin = evolve(generations=5, pop_size=7, elites=2, xover="random", mutation_rate=0.05, mutation="uniform", social_net_size=20, 
             run_duration=50, show=True, save=True, pruning_threshold=0.1, eval="clustering")

print(fin)

params = fin.iloc[pd.to_numeric(fin.fitness).idxmax()]['search']
print(params)

# Create social_net for evaluation using optimal evolved parameters
test = social_net(size=20, beliefs_size=20, tolerance=params[0], rationality=params[1], pressure=params[2], a=params[3], b=params[4], choice_type="beliefs")

edge_data = pd.DataFrame()
community_data = pd.DataFrame()
community_data['clustering'] = 0
community_data['modularity'] = 0

# test run social_net for 100 time steps and collect community measures at each step       
for i in range(100):
    test.interaction_event()
    if i%10 == 0:
        print(i)
    
    comm_row = []
    comm_row.append(global_clustering(prune_graph(test.I, type='threshold', threshold=.1)))
    comm_row.append(modularity(test.I))
    community_data.loc[i] = comm_row

# plot community measures for social_net eval run
fig = plt.figure(figsize=(24, 8))
ax = fig.add_subplot(111)
community_data.plot(lw=1, ax=ax)
plt.title('Evaluation Run', fontsize=22)
plt.xlabel('Run duration', fontsize=18)
plt.ylabel('Fitness', fontsize=18)


compare_clustering(test.I)