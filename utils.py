import numpy as np


def generate_parallel_intervals(list_num, pool_num):
    gap = list_num // pool_num
    intervals = []
    for i in range(pool_num):
        if i == pool_num - 1:
            intervals.append((i * gap, list_num))
        else:
            intervals.append((i * gap, (i+1) * gap))
    return intervals


def compute_lower_bounds_of_coverage(cov, delta=0.025):
    tmp = np.sqrt(cov + 2. * np.log(1./ delta) / 9.) - np.sqrt(np.log(1./ delta) / 2.)
    return tmp ** 2 - np.log(1./ delta) / 18.

def compute_upper_bounds_of_coverage(cov, delta=0.025):
    tmp = np.sqrt(cov / 0.405 + np.log(1./ delta) / 2.) + np.sqrt(np.log(1./ delta) / 2.)
    return tmp ** 2


if __name__ == "__main__":
    print(generate_parallel_intervals(100000, 3))
