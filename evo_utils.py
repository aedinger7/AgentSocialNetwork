import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from social_net import social_net
from utils import compare_clustering
_RUN_DURATION = 50
_BELIEFS_SIZE = 20
_SOCIAL_SIZE = 20

"""
Includes various functions for running evolutions on social_net objects, representing networks of social agents.
"""

def feval(params, run_duration=_RUN_DURATION, show=False, pruning_threshold=0, eval="clustering"):
    """Function for evaluating evolutions of social_nets.

    Args:
        params (list of floats): Parameters defining the social network to be evaluated
        run_duration ([type], optional): Defines run length for evaluation of a single network. Defaults to _RUN_DURATION.
        show (bool, optional): If True, display parameters and graphs for comparing performance under different evaluation techniques. Defaults to False.
        pruning_threshold (float in [0,1], optional): float in [0,1]: Determines whether networks are pruned of edges below the specified weight threshold before evaluation. May improve performance of some eval functions. Defaults to 0.
        eval (str, optional): Determines which evaluation function to use. Currently supports "clustering" and "modularity". Defaults to "clustering".


    Returns:
        float in [0,1]: Evaluation score of network under specified evaluation method
    """
    network = social_net(size=_SOCIAL_SIZE, beliefs_size=_BELIEFS_SIZE, rationality=params[0], pressure=params[1], tolerance=params[2], a=params[3], b=params[4])

    runs = 0
    while runs<run_duration:
        runs += 1
        network.interaction_event()
    
    if show:
        print(f"rationality={params[0]}, pressure={params[1]}, tolerance={params[2]}, a={params[3]}")
        compare_clustering(network.I, pruning_threshold=pruning_threshold)
        # network.agent_connections()

    if eval=="clustering":
        return max(0, 1-network.social_clustering(pruning_threshold=pruning_threshold))

    if eval=="modularity":
        try:
            return max(0, network.social_modularity(pruning_threshold=pruning_threshold))
        except:
            return 0

# def plot(outputs, step_size=_STEP_SIZE):
#     run_duration = transient_duration + eval_duration
#     # plot oscillator output
#     plt.plot(np.arange(0,run_duration,step_size),outputs[:,0])
#     plt.plot(np.arange(0,run_duration,step_size),outputs[:,1])
#     plt.xlabel('Time')
#     plt.ylabel('Neuron outputs')
#     plt.show()
    

def init_pop(pop_size, pruning_threshold=0, eval="clustering"):
    """Helper function for evolve function. Creates initial population to be evolved. 

    Args:
        pop_size (int): size of population to be evolved
        pruning_threshold (int, optional): Pruning parameter for network evaluations. See feval function. Defaults to 0.
        eval (str, optional): Determines which evaluation function to use. Currently supports "clustering" and "modularity". Defaults to "clustering".
    Returns:
        [DataFrame]: contains parameter sets and fitness of initial population

    """

    print("Initializing population... ")
    pop = pd.DataFrame(index=pd.RangeIndex(start=0, stop=pop_size, name="individual"), columns=["search", "fitness"])
    for i in range(pop_size):
        pop.iloc[i]["search"] = list(np.random.uniform(low=0, high=1, size=5))
        pop.iloc[i]["fitness"] = feval(pop.iloc[i]["search"], pruning_threshold=pruning_threshold, eval=eval)
    return(pop)


def next_gen(prev, elites=5, xover="random", mutation_rate=.1, run_duration=_RUN_DURATION, pruning_threshold=0, eval="clustering"):
    """ Helper function for evolve. Takes a generation of the population and generates the following generation according to specified 
        parameters.

    Args:
        prev (DataFrame): Previous generation.
        elites (int, optional): Specifies the number of best agents from previous generation to carry through. Defaults to 5.
        xover (str, optional): Determines production of children for next generation. If "random", each child takes parameters from two parents and randomly selects parameters from each.  Defaults to "random".
        mutation_rate (float, optional): [description]. Defaults to .1.
        run_duration ([type], optional): Defines run length for evaluation of a single network. Defaults to _RUN_DURATION.
        pruning_threshold (int, optional): # pruning_threshold - float in [0,1]: Determines whether networks are pruned of edges below the specified weight threshold before evaluation. May improve performance of some eval functions. Defaults to 0.
        eval (str, optional): Determines which evaluation function to use. Currently supports "clustering" and "modularity". Defaults to "clustering".

    Returns:
        [DataFrame]: contains parameter sets and fitness of new generation.
    """
    pop = pd.DataFrame(index=pd.RangeIndex(start=0, stop=len(prev), name="individual"), columns=["search", "fitness"])
    prev.sort_values("fitness", ascending=False, inplace=True)
    prev.reset_index(drop=True, inplace=True)

    for i in range(elites):
        pop.iloc[i] = prev.iloc[i].copy()
        # print("elite ", i,": ", pop)
        
    for i in range(elites, len(pop)):
        x = prev.sample(weights=prev["fitness"]).iloc[0]["search"].copy()
        
        if xover == "none":
            child=x
        if xover == "random":
            splice = np.random.randint(0,3)
            y = prev.sample(weights=prev["fitness"]).iloc[0]["search"]
            child = x[:splice] + y[splice:]

            # naive method for truncated normal distribution in range [0,1]
            for j in range(len(child)):
                param = np.random.normal(child[j], scale=mutation_rate)
                while param<0 or param>1:
                    param = np.random.normal(child[j], scale=mutation_rate)
                child[j] = param
        
        # print("child ", i, ":", child)
        pop.loc[i]["search"] = child
    
    for i in range(len(pop)):
        pop.iloc[i]["fitness"] = feval(pop.iloc[i]["search"], run_duration=run_duration, pruning_threshold=pruning_threshold, eval=eval)

    print("fin: ", pop)
    return pop

def evolve(generations=20, pop_size=10, elites=2, xover="random", mutation="uniform", mutation_rate=0.05, run_duration=_RUN_DURATION, show=False, save=True, pruning_threshold=0, eval="clustering"):
    """Carries out evolution process with parameters for specifying population, evolution, and evaluation characteristics.

    Args:
        generations (int, optional): Number of generations to evolve. Defaults to 20.
        pop_size (int, optional): Size of population for evolution. Defaults to 10.
        elites (int, optional): Specifies the number of best agents from previous generation to carry through. Defaults to 5.
        xover (str, optional): Determines production of children for next generation. If "random", each child takes parameters from two parents and randomly selects parameters from each.  Defaults to "random".
        mutation_rate (float, optional): Passed to next_gen. Rate of change (mutation) for parameters from parent to child. Defaults to .1.
        run_duration ([type], optional): Defines run length for evaluation of a single network. Defaults to _RUN_DURATION.
        pruning_threshold (int, optional): # pruning_threshold - float in [0,1]: Determines whether networks are pruned of edges below the specified weight threshold before evaluation. May improve performance of some eval functions. Defaults to 0.
        show (bool, optional): Display results and population characteristics every fifth generation. Defaults to True.
        save (bool, optional): Saves .csv files of every fifth generation. Primarily used for testing. Defaults to False.
        eval (str, optional): Passed to feval. Determines which evaluation function to use. Currently supports "clustering" and "modularity". Defaults to "clustering".
        pruning_threshold (int, optional): Passed to feval. Determines whether networks should be pruned before evaluation. May improve evaluation accuracy. Defaults to 0.
    """
    print(f"Evolving: pop size = {pop_size} for {generations} generations")
    prev = init_pop(pop_size, pruning_threshold=pruning_threshold, eval=eval)

    evos = pd.DataFrame(index=pd.RangeIndex(start=0, stop=generations, name="generation"), columns=["best", "mean"])
    if mutation == "uniform":
        for i in range(0, generations):
            print(f"Generation {i}")
            evos.loc[i]["best"] = prev["fitness"].max()
            evos.loc[i]["mean"] = prev["fitness"].mean()
#             mutation_rate=max(2-evos.loc[i]["best"]*2000,.1)
            print("Gen: ", i, "\tMax: ", evos.loc[i]["best"], "\tMean: ", evos.loc[i]["mean"])
            prev = next_gen(prev, elites=elites, mutation_rate=mutation_rate, run_duration=run_duration, pruning_threshold=pruning_threshold, eval=eval)
            
            

            if(i%5) == 0 or i==generations-1:
                if save:
                    prev.to_csv(f"evo_run_gen{i}.csv")
                if show:
                    print(f"Generation {i} best subject:", prev.iloc[pd.to_numeric(prev.fitness).idxmax()])
                    feval(prev.iloc[pd.to_numeric(prev.fitness).idxmax()]['search'], show=True, pruning_threshold=pruning_threshold, eval=eval)

#     if mutation == "non-uniform":
#         for i in range(0, generations):
#             evos.loc[i]["best"] = prev["fitness"].max()
#             evos.loc[i]["mean"] = prev["fitness"].mean()
#           
#             mutation_rate = 0.3*((256 - evos.loc[i]["mean"])/256)**2
#             if ((256 - evos.loc[i]["mean"])/256) < 0.2:
#                    
#             mutation_rate = 0.15*((256 - evos.loc[i]["mean"])/256)**2
#             prev = next_gen(prev, elites=elites, xover = xover, mutation_rate = mutation_rate)
            
    plt.figure(figsize=(50,20))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Generations', fontsize=26)
    plt.ylabel('Fitness', fontsize=26)
    plt.plot(evos)
    plt.show()
    print(evos)

    evos.to_csv(f"evo_{run_duration}_{eval}_{pruning_threshold}.csv")

    return prev