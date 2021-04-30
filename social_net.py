# pylint: disable=unused-variable
import networkx as nx
import numpy as np
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt
import random
from agent import agent
from utils import global_clustering, modularity

"""
Defines an object representing a network of social agents.
"""

class social_net:

    def __init__(self, size, beliefs_size, coherent=False, rationality=1, pressure=0, tolerance=0.7, a=0.5, b=0.1, choice_type="both"):
        """Constructor function for social_net objects.

        Args:
            size ([type]): Number of agents in social network
            beliefs_size ([type]): Number of nodes in agent belief networks
            coherent (bool, optional): If true, all agents in network begin with completely coherent internal belief set. Defaults to False.
            rationality (int, optional): preference of agents for improvement to coherence in belief networks. Defaults to 1.
            pressure (int, optional): rate of agents to accept changes to beliefs that may not improve belief coherence. Defaults to 0.
            tolerance (int, optional): probability of ending interaction after unsuccessful actions occur. Defaults to 3.
            a (float, optional): Rate of change in social preference of agent for another agent with which it interacts. Convention for a, b as in Gelardi et al 2021. Defaults to 0.5.
            b (float, optional): Rate of decay in social preference of agent for all other agents following an interaction with a specific agent. Convention for a, b as in Gelardi et al 2021. Defaults to 0.1.
            choice_type (str, optional): Parameter for determining how agents in the network select other agents for interaction. If "both", uses a combination of beliefs and social interactions. Else, uses only beliefs. Defaults to "both".
        """
        self.size = size
        self.agents = []
        self.rationality = rationality
        self.pressure = pressure
        self.tolerance = tolerance
        self.choice_type = choice_type
        self.a = a
        self.b = b
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


    def social_clustering(self, pruning_threshold=0):
        """Global clustering coefficient from Barrat et al 2004. Average clustering over all nodes in social connectivity network.

        Args:
            pruning_threshold (int, optional): Prunes all edges in graph below given value before computing. Defaults to 0.

        Returns:
            float: Clustering coefficient
        """
        return global_clustering(self.I, pruning_threshold=pruning_threshold)

    
    def belief_clustering(self, pruning_threshold=0):
        """Global clustering coefficient from Barrat et al 2004. Average clustering over all nodes in agent agreement network.

        Args:
            pruning_threshold (int, optional): Prunes all edges in graph below given value before computing. Defaults to 0.

        Returns:
            float: Clustering coefficient
        """
        return global_clustering(self.P, pruning_threshold=pruning_threshold)


    def social_modularity(self, pruning_threshold=0):
        """Computes network modularity measure for social connectivity network with Louvain method from Blondel et al 2004 and implementation from Aynaud 2009.

        Args:
            pruning_threshold (int, optional): Prunes all edges in graph below given value before computing. Defaults to 0.

        Returns:
            float: Network modularity
        """
        return modularity(self.I, pruning_threshold=pruning_threshold)

    
    def belief_modularity(self, pruning_threshold=0):
        """Computes network modularity measure for agreement network with Louvain method from Blondel et al 2004 and implementation from Aynaud 2009.

        Args:
            pruning_threshold (int, optional): Prunes all edges in graph below given value before computing. Defaults to 0.

        Returns:
            float: Network modularity
        """
        return modularity(self.P, pruning_threshold=pruning_threshold)
        
    
    def agent_coherence(self):
        """Displays coherence of all agents in the network. See agent.coherence.
        """
        for i, agent in enumerate(self.agents):
            print(i, agent.coherence())
    
    
    def agent_agreement(self):
        """Displays pairwise agreement of all agents in network. See agent.agreement. Primarily used for testing.
        """
        for edge in self.P.edges():
            print(edge, self.P.edges()[edge])
            
    
    def agent_interactions(self):
        """Displays edge weights of social connectivity network. Primarily used for testing.
        """
        for edge in self.I.edges():
            print(edge, self.I.edges()[edge])
            
    
    def agent_connections(self):
        """Displays social connectivity and agent agreement values. Primarily used for testing.
        """
        print('Edge', '\t', 'Belief Agreement', '\t\t', 'Social Preference')
        for edge in self.I.edges():
            print(edge, self.P.edges()[edge], '\t', self.I.edges()[edge])
            
    
    def __update_edges(self):
        """Helper function. Updates edges in agent agreement network after changes in internal concept networks occur.
        """
        for i, a in enumerate(self.agents):
            for j, b in enumerate(self.agents):
                if i != j:
                    self.P.edges[(i,j)]['weight'] = a.agreement(b)
                    
    
    def __interaction_choice(self, i):
        """Chooses target of interaction for an agent as a function of social and belief network values. Object variable self.choice_type 
        determines method for choosing, which may use agent agreement alone or agent agreement and social connectivity. Defaults to beliefs,
        which was used for all testing of this project.

        Args:
            i (int): Agent in social network that will choose a target agent for interaction

        Returns:
            int: Chosen agent as target for interaction.
        """
        probabilities = []
        other_agents = []
        for j in range(len(self.agents)):
            if j != i:
                if self.choice_type == "beliefs":
                    probabilities.append(self.P.edges[(i,j)]['weight'])        
                else:
                    probabilities.append(self.P.edges[(i,j)]['weight'] + self.I.edges[(i,j)]['weight'])
                other_agents.append(j)
        probabilities = [p/sum(probabilities) for p in probabilities]
        c = np.random.choice(other_agents, p=probabilities)
        return c
        
        
    def interaction_event(self):
        """Execute a series of interactions in the network, where each agent chooses a target agent for interaction, and updates
        the social connectivity network accordingly.
        """
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
    
    