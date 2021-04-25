import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.special
from utils import network_coherence, coherify

class agent:
    # Agent constructor method
    # size: int - size of agent belief network
    # coherent, weighted, zeros: booleans - determine belief network characteristics (see __random_graph method)
    def __init__(self, size, coherent=False, weighted=False, zeros=False):
        self.size = size
        self.beliefs = agent.__random_graph(size, coherent=coherent, weighted=weighted, zeros=zeros)
        
        # Additional params:
        # Tolerance - length of interactions
        # Plasticity - Acceptance rate for all edges?
        #    vs. 
        # Zealotry - Resistance to changes to belief networks
        # Rationality - Acceptance rate for coherence-improving edges
    
    
    # Returns random graph of given size for agent construction, defaults to complete coherent graph
    #  with weights limited to [-1, 1]
    # coherent: boolean - returns random coherent graph if true, random incoherent otherwise
    # weighted: boolean - connection weights are float values in [-1, 1] if true
    # zeros: boolean - connection weights are int values in [-1, 0, 1] if true
    @staticmethod
    def __random_graph(size, coherent=True, weighted=False, zeros=False):
        G = nx.complete_graph(size)
        
        if weighted == True:
            for (u, v) in G.edges():
                G.edges[u,v]['weight'] = np.random.uniform(low=-1, high=1)
        
        elif zeros == True:
            p=np.random.uniform(low=0, high=1, size=3)
            s = sum(p)
            p=[i/s for i in p]
            for (u, v) in G.edges():
                G.edges[u,v]['weight'] = np.random.choice([-1,0,1], p=p)
            
        else:
            dist = np.random.uniform(low=0, high=1)
            for (u, v) in G.edges():
                G.edges[u,v]['weight'] = np.random.choice([-1,1], p=[dist,1-dist])

        if coherent == True:
            coherify(G)

        return G
    
    
    def coherence(self):
        G = self.beliefs
        return network_coherence(G)
        
        
    # Returns number of agreeing edges between networks G1 & G2
    def agreement(self, other):
        d = 0
        count = 0
        if self.size == other.size:
            for [j,k] in self.beliefs.edges:
                if self.beliefs.edges[j,k]['weight']*other.beliefs.edges[j,k]['weight'] > 0:
                    d += 1
                if self.beliefs.edges[j,k]['weight']*other.beliefs.edges[j,k]['weight'] != 0:
                    count += 1
            return d/len(self.beliefs.edges)
        else: 
            return False
        
        
    # Compare a candidate edge weight with current edge weight in network, substitute weight if candidate edge 
    #  improves network coherence
    # Returns True if substitution occurs, False otherwise
    def substitute(self, j, k, w, rationality=1, pressure=0):
        G = self.beliefs.copy()
        G.edges[j,k]['weight'] = w
        if network_coherence(G) > self.coherence() and np.random.random() < rationality:
            self.beliefs.edges[j,k]['weight'] = w
            return True
        
        if network_coherence(G) == self.coherence():
            return True
            
        elif np.random.random() < pressure:
            self.beliefs.edges[j,k]['weight'] = w
            return True
            
        else:
            return False
        
    
    # Print list of graph edges and weights and edge weight matrix
    def disp(self):
        for edge in self.beliefs.edges():
            print(edge, self.beliefs.edges()[edge])
        print(nx.to_numpy_array(self.beliefs))
        
    
    # Carries out an agent-agent interaction as a series of choices of edges in belief network to compare and
    #  attempt to integrate into other agents' belief network 
    # Other - other agent as target of interaction
    # Returns value between 0 and 1 representing percentage of successful actions taken
    def interaction(self, other, tolerance=0.3, rationality=1, pressure=0):
        d = 0
        count = 1
        n1 = np.random.choice(range(self.size))
        n2 = n1
        while n2 == n1:
            n2 = np.random.choice(range(self.size))


        while count < self.size:
            n1 = n2
            while n2 == n1:
                n2 = np.random.choice(range(self.size))
            if not other.substitute(n1, n2, self.beliefs.edges[n1,n2]['weight'], rationality=rationality, pressure=pressure):
                d += 1
                if np.random.random() < tolerance:
                    break
                
            
            count += 1

            n1 = n2
            while n2 == n1:
                n2 = np.random.choice(range(self.size))
            if not self.substitute(n1, n2, other.beliefs.edges[n1,n2]['weight'], rationality=rationality, pressure=pressure):
                d += 1
                if np.random.random() < tolerance:
                    break

            count += 1

        return (count-d)/count
    
    




