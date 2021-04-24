import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from social_net import social_net
_RUN_DURATION = 50
_BELIEFS_SIZE = 20
_SOCIAL_SIZE = 20

# def run(params, run_duration=_RUN_DURATION):
#     net_size = 2
    
#     # set up network
#     network = social_net()

#     return np.asarray(outputs)
    
def feval(params, run_duration=_RUN_DURATION, show=False):
    # set up network
    network = social_net(size=_SOCIAL_SIZE, beliefs_size=_BELIEFS_SIZE, rationality=params[0], pressure=params[1], tolerance=params[2], a=params[3])

    runs = 0
    while runs<_RUN_DURATION:
        runs += 1
        network.interaction_event()
    
    if show:
        print(f"rationality={params[0]}, pressure={params[1]}, tolerance={params[2]}, a={params[3]}")
        network.agent_connections()
        
    return network.clustering_coeff()

# def plot(outputs, step_size=_STEP_SIZE):
#     run_duration = transient_duration + eval_duration
#     # plot oscillator output
#     plt.plot(np.arange(0,run_duration,step_size),outputs[:,0])
#     plt.plot(np.arange(0,run_duration,step_size),outputs[:,1])
#     plt.xlabel('Time')
#     plt.ylabel('Neuron outputs')
#     plt.show()
    

def init_pop(pop_size):
    print("Initializing population... ")
    pop = pd.DataFrame(index=pd.RangeIndex(start=0, stop=pop_size, name="individual"), columns=["search", "fitness"])
    for i in range(pop_size):
        pop.iloc[i]["search"] = list(np.random.uniform(low=0, high=1, size=4))
        print(i, pop.iloc[i]["search"])
        pop.iloc[i]["fitness"] = feval(pop.iloc[i]["search"])
    return(pop)

def next_gen(prev, elites=5, xover="random", mutation_rate=.1):
    pop = pd.DataFrame(index=pd.RangeIndex(start=0, stop=len(prev), name="individual"), columns=["search", "fitness"])
    prev.sort_values("fitness", ascending=False, inplace=True)
    prev.reset_index(drop=True, inplace=True)

    print("a:", prev)
    
    for i in range(elites):
        pop.iloc[i]['search'] = prev.iloc[i]['search'].copy()
        
    for i in range(elites, len(pop)):
        x = prev.sample(weights=prev["fitness"]).iloc[0]["search"].copy()
        
        if xover == "none":
            child=x
        if xover == "random":
            splice = np.random.randint(0,3)
            y = prev.sample(weights=prev["fitness"]).iloc[0]["search"]
            child = x[:splice] + y[splice:]

            # naive method for truncated normal distribution in range [0,1]
            for i in range(len(child)):
                param = np.random.normal(child[i], scale=mutation_rate)
                while param<0 or param>1:
                    param = np.random.normal(child[i], scale=mutation_rate)
                child[i] = param
        
        pop.loc[i]["search"] = child

    print("b:", prev)
    
    
    for i in range(len(pop)):
        # try:  
        #     print(i)
        pop.iloc[i]["fitness"] = feval(pop.iloc[i]["search"])
        # except Exception as e: 
        #     print(e)
        #     params = pop.iloc[i]['search']
        #     print(f"Error occurred with parameter set: {params}")
    
    return pop

def evolve(generations=20, pop_size=10, elites=2, xover="random", mutation="uniform"):
    print(f"Evolving: pop size = {pop_size} for {generations} generations")
    prev = init_pop(pop_size)

    print("c: ", prev)

    evos = pd.DataFrame(index=pd.RangeIndex(start=0, stop=generations, name="generation"), columns=["best", "mean"])
    if mutation == "uniform":
        for i in range(0, generations):
            evos.loc[i]["best"] = prev["fitness"].max()
            evos.loc[i]["mean"] = prev["fitness"].mean()
#             mutation_rate=max(2-evos.loc[i]["best"]*2000,.1)
            mutation_rate=2
            prev.to_csv("evo_test_run_m1.csv")
            prev = next_gen(prev, elites=elites, mutation_rate=mutation_rate)
            print("Gen: ", i, "\tMax: ", evos.loc[i]["best"], "\tMean: ", evos.loc[i]["mean"])
            
            prev.to_csv("evo_test_run.csv")

            if(i%5) == 0:
                print(prev.iloc[prev['fitness'].idxmax()]['search'])
                print(f"Generation {i} best subject:", prev.iloc[prev['fitness'].idxmax()])
                feval(prev.iloc[prev['fitness'].idxmax()]['search'], show=True)

            
            
#     if mutation == "non-uniform":
#         for i in range(0, generations):
#             evos.loc[i]["best"] = prev["fitness"].max()
#             evos.loc[i]["mean"] = prev["fitness"].mean()
            
#             mutation_rate = 0.3*((256 - evos.loc[i]["mean"])/256)**2
#             if ((256 - evos.loc[i]["mean"])/256) < 0.2:
                    
            
#             mutation_rate = 0.15*((256 - evos.loc[i]["mean"])/256)**2
#             prev = next_gen(prev, elites=elites, xover = xover, mutation_rate = mutation_rate)
            
            
            
    plt.plot(evos)
    plt.show()
    print(evos)

    return prev