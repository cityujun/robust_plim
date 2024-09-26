import numpy as np

def func_rand_assign(infile, num_nodes, seed=128):
    outfile = infile.replace('.txt', '_type.txt')
    np.random.seed(seed)
	
    # {1: 2x-x^2, 2: x, 3: x^2}
    # assignment 1 -> 1: 65%, 2: 20%, 3: 15%
    prob = [0.65, 0.20, 0.15]
    rand_assign = np.random.choice(3, num_nodes, p=prob)
    with open(outfile, 'w') as fout:
        for nodeID, assign in zip(range(num_nodes), rand_assign):
            print(nodeID, assign+1, file=fout)


def graph_reform(infile, bidirect=False):
    outfile = infile.replace('.txt', '_degree.txt')
    fin = open(infile, 'r')
    all_lines = fin.readlines()
    
    skip_lines = 0
    dict_nodes = {}
    node_idx = 0
    for line in all_lines:
        if line.startswith('#'):
            skip_lines += 1
            continue
        fromnode, tonode = map(int, line.strip().split())
        
        if fromnode not in dict_nodes:
            dict_nodes[fromnode] = [node_idx, 0]
            node_idx += 1
        if tonode not in dict_nodes:
            dict_nodes[tonode] = [node_idx, 0]
            node_idx += 1
        
        dict_nodes[tonode][1] += 1
        if bidirect:
            dict_nodes[fromnode][1] += 1
    
    print(f'Skip {skip_lines} lines.')
    
    with open(outfile, 'w') as fout:
        for line in all_lines:
            if line.startswith('#'):
                continue
            fromnode, tonode = map(int, line.strip().split())
            if bidirect:
                print(dict_nodes[fromnode][0], dict_nodes[tonode][0], dict_nodes[fromnode][1], dict_nodes[tonode][1], file=fout)
            else:
                print(dict_nodes[fromnode][0], dict_nodes[tonode][0], dict_nodes[tonode][1], file=fout)

    fin.close()

if __name__ == '__main__':
    import os, sys

    dataset = sys.argv[1]
    assert dataset in ['wiki', 'amazon', 'google', 'astroph', 'hepph']
    args_dict = {
        'wiki': ('wiki-Vote.txt', 7115, False),
        'amazon': ('com-amazon.ungraph.txt', 334863, True),
        'google': ('web-Google.txt', 875713, False),
        'astroph': ('ca-AstroPh.txt', 18772, False),
        'hepph': ('ca-HepPh.txt', 12008, False),
    }
    graph_reform(args_dict[dataset][0], args_dict[dataset][2])
    func_rand_assign(args_dict[dataset][0], args_dict[dataset][1])
