import sys
import time
from copy import deepcopy
# import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from base import UtilityTriple, PriorityQueue, Activator
from utils import *


class GreedyAlgorithm:
    def __init__(self, graph, cost_unit):
        self.graph = graph
        self.n_V = graph.n_V
        self.node2type = self.graph.load_node_type()
        assert cost_unit == graph.cost_unit
        self.activator = Activator(cost_unit, graph.eta, graph.n_E)
        self.cost_pool = [round((i+1) * cost_unit, 2) for i in range(int(1. / cost_unit))]
        # print('cost pool: ', self.cost_pool)
    
    def reset_sampling(self):
        self.activator.reset_sampling()
        self.graph.rev_adj.reset_sampling()
    
    @staticmethod
    def _compute_probed_coverage(rr_sets, probs):
        res = 0
        seeds = set([i for (i, p) in enumerate(probs) if p > 0])
        for rr_set in rr_sets:
            common_seeds = seeds.intersection(rr_set)
            if not common_seeds:
                continue
            tmp_res = 1
            for node_idx in common_seeds:
                tmp_res *= (1 - probs[node_idx])
            res += 1 - tmp_res
        return res
    
    @staticmethod
    def _generate_node_coverage(rr_sets, n_V):
        ## alternative method to generate node2coverage, considering the sparsity of rr_sets, super faster
        st = time.time()
        node2coverage = [0] * n_V
        for rr_set in rr_sets:
            for node_idx in rr_set:
                node2coverage[node_idx] += 1
        et = time.time()
        print(f'Spend {round(et - st, 1)}s generating node2coverage.')
        return node2coverage
    
    def estimate_spread(self, costs, rr_sets, act_probs):
        # C -> P
        probs = [act_probs[(t, c)] if c > 0 else 0 for (t, c) in zip(self.node2type, costs)]
        # P -> estimate of influence spread, intersection

        res = self._compute_probed_coverage(rr_sets, probs)
        return res * self.n_V / len(rr_sets)
    
    def estimate_spread_bounds(self, costs, rr_sets, eval_rr_sets): # only for true rev_adj and act_probs
        probs = [self.activator.true_act_probs[(t, c)] if c > 0 else 0 for (t, c) in zip(self.node2type, costs)]
        assert len(eval_rr_sets) == len(rr_sets)
        coverage = self._compute_probed_coverage(rr_sets, probs)
        eval_coverage = self._compute_probed_coverage(eval_rr_sets, probs)
        lo_bound = compute_lower_bounds_of_coverage(eval_coverage, 1. / self.n_V)
        up_bound = compute_upper_bounds_of_coverage(coverage, 1. / self.n_V)
        
        return lo_bound, up_bound
    
    def initialize_queue(self, rr_sets, act_probs):
        node2coverage = self._generate_node_coverage(rr_sets, self.n_V)
        node2coverage = [num_cover * self.n_V / len(rr_sets) for num_cover in node2coverage]
        
        utility_queue = PriorityQueue()
        for node_idx in range(self.n_V):
            for cost in self.cost_pool:
                prob = act_probs[(self.node2type[node_idx], cost)] # 0 otherwise
                val = node2coverage[node_idx] * prob / cost
                utility_queue.put(UtilityTriple(node_idx, cost, val))
        
        return utility_queue

    def _vanilla_greedy(self, budget, rr_sets, act_probs):
        print('Initializing priority queue...')
        st = time.time()
        utility_queue = self.initialize_queue(rr_sets, act_probs)
        et = time.time()
        # print(utility_queue.qsize())
        
        opt_costs = [0.] * self.n_V
        budget_left = budget
        print('Starting greedy loop...')
        while not utility_queue.empty():
            triple = utility_queue.get()

            cur_cost = opt_costs[triple.index]
            if  cur_cost >= triple.cost or budget_left + cur_cost - triple.cost < 0:
                continue

            ## lazy greedy
            cur_val = self.estimate_spread(opt_costs, rr_sets, act_probs)
            tmp_costs = deepcopy(opt_costs)
            tmp_costs[triple.index] = triple.cost
            new_val = self.estimate_spread(tmp_costs, rr_sets, act_probs)
            gain = (new_val - cur_val) / triple.cost

            if gain >= utility_queue.top_priority():
                budget_left -= (triple.cost - cur_cost)
                opt_costs[triple.index] = triple.cost
            else:
                utility_queue.put(UtilityTriple(triple.index, triple.cost, gain))

        print(f'Greedy algorithm finished. Spending {round(et - st, 1)}s initializing and {round(time.time() - et, 1)}s in loop.')
        sys.stdout.flush()
        return opt_costs
    
    def _vanilla_greedy_multiple(self, args):
        budget, rr_sets, act_probs = args
        return self._vanilla_greedy(budget, rr_sets, act_probs)
    
    def vanilla_greedy_with_true_param(self, num_rr_sets, budget_list):
        all_rr_sets = self.graph.rr_simulation(num_rr_sets * 2)
        rr_sets, eval_rr_sets = all_rr_sets[:num_rr_sets], all_rr_sets[num_rr_sets:]
        
        param_list = [(budget, rr_sets, self.activator.true_act_probs) for budget in budget_list]
        pool = Pool(len(budget_list))
        pool_res = pool.map(self._vanilla_greedy_multiple, param_list)
        pool.close()
        pool.join()
        
        ret_ratios = []
        for opt_costs, budget in zip(pool_res, budget_list):
            lo, up = self.estimate_spread_bounds(opt_costs, rr_sets, eval_rr_sets)
            ret_ratios.append(round(lo / up, 5))
            print(budget, round(lo / up, 5), lo, up)
            print('\n##################')
            sys.stdout.flush()
        return ret_ratios
    
    def lu_greedy_with_width_or_samples(self, num_rr_sets, budget, width, n_samples):
        pool = Pool(2)
        self.activator.update_act_probs(width, n_samples)
        lo_rr_sets, up_rr_sets = self.graph.rr_simulation_lo_up(num_rr_sets, width, n_samples)
        
        # lower_costs = self._vanilla_greedy(budget, lo_rr_sets, self.activator.lo_act_probs)
        # upper_costs = self._vanilla_greedy(budget, up_rr_sets, self.activator.up_act_probs)
        param_list = [(budget, lo_rr_sets, self.activator.lo_act_probs), (budget, up_rr_sets, self.activator.up_act_probs)]
        pool_res = pool.map(self._vanilla_greedy_multiple, param_list)
        pool.close()
        pool.join()
        lower_costs, upper_costs = pool_res[0], pool_res[1]

        # lo_rr_sets, up_rr_sets = self.graph.rr_simulation_lo_up(num_rr_sets, width, n_samples)
        lower_val = self.estimate_spread(lower_costs, lo_rr_sets, self.activator.lo_act_probs)
        upper_val = self.estimate_spread(upper_costs, lo_rr_sets, self.activator.lo_act_probs)
        critic_val = self.estimate_spread(upper_costs, up_rr_sets, self.activator.up_act_probs)
        opt_val = max(lower_val, upper_val)
        print(round(opt_val / critic_val, 5), opt_val, critic_val)
        print('\n##################')
        return round(opt_val / critic_val, 5), opt_val, critic_val
