import sys
import numpy as np
import random
import time
from tqdm import tqdm
# from collections import defaultdict
from base.node import Node


class revAdjacency:
    def __init__(self, graph_file, ratio, n_V, n_E, cost_unit, eta):
        self.graph_file = graph_file
        self.ratio = ratio
        self.n_V = n_V
        self.n_E = n_E
        self.eta = eta
        self.cost_unit = cost_unit
        # self.matrix = self.load_true_matrix()
    
    def load_true_matrix(self):
        matrix = [[] for _ in range(self.n_V)]

        st = time.time()
        with open(self.graph_file, 'r') as fin:
            for line in fin.readlines():
                try: # undirected graph
                    u, v, u_in_deg, v_in_deg = map(int, line.strip().split())
                    prop_prob = self.ratio / u_in_deg
                    rev_node = Node(v, prop_prob)
                    matrix[u].append(rev_node)
                except: # directed
                    u, v, v_in_deg = map(int, line.strip().split())
                prop_prob = self.ratio / v_in_deg
                rev_node = Node(u, prop_prob)
                matrix[v].append(rev_node)
        print(f'Graph loaded. Time consumed {round(time.time() - st, 1)}s.')
        return matrix
    
    def reset_sampling(self):
        self.total_n_samples = 0
        self.true_matrix = self.load_true_matrix()
        self.sample_freqs = [[0.] * len(nodes) for u, nodes in enumerate(self.true_matrix)]
        n_E = sum([len(node_list) for node_list in self.true_matrix])
        assert n_E == self.n_E

    def update_matrix(self, width=0, n_samples=0):
        if width > 0:
            assert n_samples == 0
            lo_matrix = self.update_with_fixed_width_lo(width)
            up_matrix = self.update_with_fixed_width_up(width)
            return lo_matrix, up_matrix
        else:
            assert n_samples > 0
            return self.update_with_samples(n_samples)
    
    def update_with_fixed_width_lo(self, width):
        true_matrix = self.load_true_matrix()
        new_matrix = [[] for _ in range(self.n_V)]
        for u, nodes in enumerate(true_matrix):
            for node in nodes:
                new_pp = max(node.pp - width / 2., 0.) 
                new_matrix[u].append(Node(node.index, new_pp))
        return new_matrix
    
    def update_with_fixed_width_up(self, width):
        true_matrix = self.load_true_matrix()
        new_matrix = [[] for _ in range(self.n_V)]
        for u, nodes in enumerate(true_matrix):
            for node in nodes:
                new_pp = min(node.pp + width / 2., 1.)
                new_matrix[u].append(Node(node.index, new_pp))
        return new_matrix

    def update_with_samples(self, n_samples):
        self.total_n_samples += n_samples
        st = time.time()
        ## update self.sample_freqs
        for u, nodes in enumerate(self.true_matrix):
            for ii, node in enumerate(nodes):
                self.sample_freqs[u][ii] += sum([1 for _ in range(n_samples) if random.random() <= node.pp])

        ## From Lemma 7 in RIM (wei chen, et al.)
        num_params = 3 * (int(1. / self.cost_unit) - 1) + self.n_E
        delta = np.sqrt((3 * np.log(2 * num_params / self.eta)) / self.total_n_samples)

        lo_matrix, up_matrix = [[] for _ in range(self.n_V)], [[] for _ in range(self.n_V)]
        upper_width = []
        for u, nodes in enumerate(self.true_matrix):
            for ii, node in enumerate(nodes):
                pp_prob = self.sample_freqs[u][ii] / self.total_n_samples
                bias = delta * np.sqrt(delta ** 2 / 4. + pp_prob)
                lo_pp_prob = max(pp_prob + delta ** 2 / 2. - bias, 0.)
                up_pp_prob = min(pp_prob + delta ** 2 / 2. + bias, 1.)
                lo_matrix[u].append(Node(node.index, lo_pp_prob))
                up_matrix[u].append(Node(node.index, up_pp_prob))
                upper_width.append(delta ** 2 / 2. + bias)
        
        print(f'Time consumed {round(time.time() - st, 1)}s, the value of delta is {round(delta, 3)}.')
        print(f'Add {n_samples} samples, number of samples for each parameter so far: {self.total_n_samples}.') 
        print(f'The mean value of upper width is {round(np.mean(upper_width), 3)}, the maximal width is {round(np.max(upper_width), 3)}')   
        return lo_matrix, up_matrix
