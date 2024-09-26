import random
import time
import numpy as np
from collections import defaultdict


class Activator:
    def __init__(self, cost_unit, eta, n_E):
        self.cost_range = [round(i * cost_unit, 2) for i in range(int(1. / cost_unit) + 1)]
        self.eta = eta
        self.n_E = n_E
        self._generate_true_act_probs()
        self.lo_act_probs, self.up_act_probs = {}, {}
    
    @staticmethod
    def prob_curve(t, c):
        if t == 3:
            return c * c
        elif t == 2:
            return c
        else:
            assert t == 1
            return (2 - c) * c
    
    def _generate_true_act_probs(self):
        self.true_act_probs = {}
        for t in range(3):
            for c in self.cost_range:
                self.true_act_probs[(t+1, c)] = self.prob_curve(t+1, c)
    
    def reset_sampling(self):
        self.total_n_samples = 0
        self.sample_freqs = defaultdict(float)
    
    def update_act_probs(self, width=0, n_samples=0):
        if width > 0:
            assert n_samples == 0
            self._update_probs_with_fixed_width(width)
        elif n_samples > 0:
            assert width == 0
            self._update_probs_with_samples(n_samples)
    
    def _update_probs_with_fixed_width(self, width):
        self.lo_act_probs, self.up_act_probs = {}, {}
        for k, v in self.true_act_probs.items():
            self.lo_act_probs[k] = max(v - width / 2., 0.)
            self.up_act_probs[k] = min(v + width / 2., 1.)
            ## boundary assumption
            if k[1] == 0.:
                self.up_act_probs[k] = 0.
            if k[1] == 1.:
                self.lo_act_probs[k] = 1.

    def _update_probs_with_samples(self, n_samples):
        self.total_n_samples += n_samples
        # st = time.time()
        for k, v in self.true_act_probs.items(): # 3*(1./cost_unit)
            self.sample_freqs[k] += sum([1 for _ in range(n_samples) if random.random() <= v])
        # sample_probs = {k: v / self.total_n_samples for k, v in self.sample_freqs.items()}
        
        ## From Lemma 7 in RIM (wei chen, et al.)
        num_params = 3 * (len(self.cost_range)-1) + self.n_E
        delta = np.sqrt((3 * np.log(2 * num_params / self.eta)) / self.total_n_samples)
        self.lo_act_probs, self.up_act_probs = {}, {}
        for key, freq in self.sample_freqs.items():
            estimated_prob = freq / self.total_n_samples
            bias = delta * np.sqrt(delta ** 2 / 4. + estimated_prob)
            self.lo_act_probs[key] = max(estimated_prob + delta ** 2 / 2. - bias, 0.)
            self.up_act_probs[key] = min(estimated_prob + delta ** 2 / 2. + bias, 1.)
            ## boundary assumption
            if key[1] == 0.:
                self.up_act_probs[k] = 0.
            if key[1] == 1.:
                self.lo_act_probs[k] = 1.
        print(f'Add {n_samples} samples, number of samples for each parameter so far: {self.total_n_samples}. The value of delta is {round(delta, 3)}.')
        
        ## rearrangement
        for t in range(1, 4, 1):
            lo = [self.lo_act_probs[(t, c)] for c in self.cost_range]
            lo.sort()
            up = [self.up_act_probs[(t, c)] for c in self.cost_range]
            up.sort()
            for i, c in enumerate(self.cost_range):
                self.lo_act_probs[(t, c)] = lo[i]
                self.up_act_probs[(t, c)] = up[i]
        
        return self.lo_act_probs, self.up_act_probs
