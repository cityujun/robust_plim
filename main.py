import os, sys
import time
from tqdm import tqdm
import numpy as np

from base import Graph
from algorithm import GreedyAlgorithm
from config import args


dataset, func = sys.argv[1], sys.argv[2]
assert dataset in ['wiki', 'amazon', 'google', 'hepph', 'astroph']
graph = Graph(*args[dataset]['graph_args'])
algo = GreedyAlgorithm(graph, cost_unit=0.1)
sys.stdout.flush()


def run_diff_num_sets_and_budget():
    budget_list = [10, 50, 100]
    num_list_map = {
        "wiki": [20000, 40000, 80000, 160000, 320000, 640000],
        "astroph": [20000, 40000, 80000, 160000, 320000, 640000],
        "hepph": [20000, 40000, 80000, 160000, 320000, 640000],
        "amazon": [200000, 400000, 800000, 1600000, 3200000, 6400000],
        "google": [200000, 400000, 800000, 1600000, 3200000, 6400000],
    }

    res = [[] for _ in range(len(budget_list))]
    for num in num_list_map[dataset]:
        # print(f'Current budget is {algo.budget}, number of RR sets is {num}')
        print(f'Current number of RR sets is {num}')
        
        ratios = algo.vanilla_greedy_with_true_param(num, budget_list)
        for i, ratio in enumerate(ratios):
            res[i].append(ratio)
        
    for b, elem in zip(budget_list, res):
        print(b, elem)


rr_num_map = {
    ("wiki", 10): 640000, ("wiki", 50): 320000, ("wiki", 100): 160000,
    ("astroph", 10): 320000, ("astroph", 50): 320000, ("astroph", 100): 160000,
    ("hepph", 10): 320000, ("hepph", 50): 320000, ("hepph", 100): 160000,
    ("amazon", 10): 6400000, ("amazon", 50): 3200000, ("amazon", 100): 1600000,
    ("google", 10): 6400000, ("google", 50): 3200000, ("google", 100): 3200000,
}


def run_diff_width_and_budget():
    budget_list = [10, 50, 100]
    width_list = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    
    for budget in budget_list:
        res = []
        for width in width_list:
            print(f'Current budget is: {budget}, the width of intervals is: {width}')
            ratio, _, _ = algo.lu_greedy_with_width_or_samples(rr_num_map[(dataset, budget)], budget, width, 0)
            res.append(ratio)
        
        print(budget, width_list)
        print(budget, res)
        print('\n##################')


def run_diff_samples_and_budget():
    budget_list = [10, 50, 100]
    num_samples_list = [10000] * 10

    for budget in budget_list:
        res = []
        algo.reset_sampling()
        for n_samples in num_samples_list:
            print(f'Current budget is {budget}, incremental number of samples is {n_samples}')
            ratio, _, _ = algo.lu_greedy_with_width_or_samples(rr_num_map[(dataset, budget)], budget, 0, n_samples)
            res.append(ratio)
        
        print(budget, np.cumsum(num_samples_list))
        print(budget, res)
        print('\n##################')


if func == 'exp1':
    for kk in range(3):
        print(f'Run {kk+1}')
        print('\n##################')
        run_diff_num_sets_and_budget()
        print('\n##################')
if func == 'exp2':
    for kk in range(3):
        print(f'Run {kk+1}')
        print('\n##################')
        run_diff_width_and_budget()
        print('\n##################')
if func == 'exp3':
    for kk in range(3):
        print(f'Run {kk+1}')
        print('\n##################')
        run_diff_samples_and_budget()
        print('\n##################')


## python -m main wiki exp1