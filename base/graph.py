import sys
import random
import time
from collections import deque
import numpy as np
# from dataclasses import dataclass
from multiprocessing import Pool

from base.adjacency import revAdjacency
from utils import *


class Graph:
    def __init__(self,
                 n_V,
                 n_E,
                 graph_file,
                 type_file,
                 ratio,
                 cost_unit,
                 eta,
        ):
        self.n_V = n_V
        self.n_E = n_E
        self.rev_adj = revAdjacency(graph_file, ratio, n_V, n_E, cost_unit, eta)
        self.cost_unit = cost_unit
        self.type_file = type_file
        self.eta = eta
    
    def load_node_type(self):
        from collections import defaultdict

        node2type = [0.] * self.n_V
        type_stat = defaultdict(int)

        st = time.time()
        with open(self.type_file, 'r') as fin:
            for line in fin.readlines():
                node_id, type_id = map(int, line.strip().split())
                node2type[node_id] = type_id
                type_stat[type_id] += 1
        
        type_stat = [(tid, round(num / self.n_V, 3)) for tid, num in type_stat.items()]
        print(f'Node type loaded. Time consumed {round(time.time() - st, 1)}s. Type ratio is ', type_stat)
        return node2type
    
    def _reverse_simulation_once(self, matrix):
        '''generate one reverse reachable set, ic model
        '''
        res = set()
        u = random.randint(0, self.n_V-1)

        que = deque()
        que.append(u)
        res.add(u)

        while que:
            v = que.popleft()
            for node in matrix[v]:
                w = node.index
                pp = node.pp

                if w not in res and random.random() <= pp:
                    res.add(w)
                    que.append(w)

        return res
    
    def _reverse_simulation(self, num, matrix):
        return [self._reverse_simulation_once(matrix) for _ in range(num)]
    
    def rr_simulation(self, num_rr_sets, matrix=None, pool_num=1):
        # assert num_rr_sets % pool_num == 0
        assert pool_num == 1
        print('RR simulating...')
        if not matrix:
            matrix = self.rev_adj.load_true_matrix()
        st = time.time()

        if pool_num > 1:
            pool = Pool(pool_num)
            pool_res = pool.map(self._reverse_simulation, [(int(num_rr_sets / pool_num), matrix)] * pool_num)
            pool.close()
            pool.join()
            rr_sets = []
            for res in pool_res:
                rr_sets += res
        else:
            rr_sets = self._reverse_simulation(num_rr_sets, matrix)

        print(f'Spend {round(time.time() - st, 1)}s simulating {num_rr_sets} RR sets.')
        sys.stdout.flush()
        return rr_sets
    
    def rr_simulation_lo_up(self, num_rr_sets, width=0, n_samples=0):
        ## update rev_adj with lower bounds
        lo_matrix, up_matrix = self.rev_adj.update_matrix(width, n_samples)
        lo_rr_sets = self.rr_simulation(num_rr_sets, lo_matrix)
        up_rr_sets = self.rr_simulation(num_rr_sets, up_matrix)
        return lo_rr_sets, up_rr_sets
