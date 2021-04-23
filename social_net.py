import networkx as nx
import numpy as np
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt
import random
from agent import agent
from utils import global_clustering

class social_net:
    # size: int - number of agents in social net
    # beliefs size: int - number of nodes in agent belief networks
    # rationality: float [0,1] - preference of agents for improvement to coherence in belief networks
    # pressure: float [0,1] - rate of agents to accept changes to beliefs that may not improve belief coherence
    # tolerance: float (0, 1) - probability of ending interaction after unsuccessful actions occur
    # a: float (0, 1) - rate of change in social preference for another agent with which it interacts
    # b: float (0, 1) - rate of decay in social preference for all other agents following an interaction with a specific agent
    # (b is currently set to be proportional to a)
    def __init__(self, size, beliefs_size, coherent=False, rationality=1, pressure=0, tolerance=3, a = 0.5):
        self.size = size
        self.agents = []
        self.rationality = rationality
        self.pressure = pressure
        self.tolerance = tolerance
        self.a = a
        self.b = (1-a/size)
        for i in range(size):
            self.agents.append(agent(size=beliefs_size, coherent=coherent, zeros=True))
        
        # Network weighted by agreement between agent belief networks
        self.P = nx.complete_graph(size)
        
        # Network weighted by past social interactions between agents
        self.I = nx.complete_graph(size)
        
        for (u, v) in self.I.edges():
            self.I.edges[u,v]['weight'] = 0
        
        # for i in self.agents:
        #     print(i)
            
        self.__update_edges()


    def clustering_coeff(self):
        return global_clustering(self.I)
        
    
    def agent_coherence(self):
        for i, agent in enumerate(self.agents):
            print(i, agent.coherence())
    
    
    def agent_agreement(self):
        for edge in self.P.edges():
            print(edge, self.P.edges()[edge])
            
    
    def agent_interactions(self):
        for edge in self.I.edges():
            print(edge, self.I.edges()[edge])
            
    
    def agent_connections(self):
        print('Edge', '\t', 'Belief Agreement', '\t\t', 'Social Preference')
        for edge in self.I.edges():
            print(edge, self.P.edges()[edge], '\t', self.I.edges()[edge])
            
    
    def __update_edges(self):
        for i, a in enumerate(self.agents):
            for j, b in enumerate(self.agents):
                if i != j:
                    self.P.edges[(i,j)]['weight'] = a.agreement(b)
                    
    
    # Chooses target of interaction for an agent as a function of social and belief network values
    def __interaction_choice(self, i):
        probabilities = []
        other_agents = []
        for j in range(len(self.agents)):
            if j != i:
                probabilities.append(self.P.edges[(i,j)]['weight'] + self.I.edges[(i,j)]['weight'])
                other_agents.append(j)
        probabilities = [p/sum(probabilities) for p in probabilities]
        c = np.random.choice(other_agents, p=probabilities)
        return c
        
        
    def interaction_event(self):
        for i, agent in enumerate(self.agents):
            choice = self.__interaction_choice(i)
            other_agent = self.agents[choice]
            if not self.I.edges[(i, choice)]:
                self.I.edges[(i, choice)]['weight'] = 0
                
            # difference between current social preference and success of interaction
            delta = (agent.interaction(other_agent, rationality=self.rationality, pressure=self.pressure,
                                        tolerance=self.tolerance) - self.I.edges[(i, choice)]['weight'])
                
            # Adjust weight of social preference of acting agent for target agent depending on success of interaction
            self.I.edges[(i, choice)]['weight'] += self.a*delta
            
            # Adjust weight of social preference of acting agent for all other agents
            for j, other_agent in enumerate(self.agents):
                if j != choice and j != i:
                    self.I.edges[(i,j)]['weight'] *= self.b
        
        self.__update_edges()
    
    