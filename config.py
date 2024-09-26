args = {
    "wiki": {
        "graph_args": [
            7115, 103689,
            'data/wiki-Vote_degree.txt',
            'data/wiki-Vote_type.txt',
            0.85,
            0.1, # cost_unit
            0.05, # parameter probability, 1 - eta
        ],
    },

    "amazon": {
        "graph_args": [
            334863, 1851744,
            'data/com-amazon.ungraph_degree.txt',
            'data/com-amazon.ungraph_type.txt',
            0.85,
            0.1,
            0.05,
        ],
    },

    "google": {
        "graph_args": [
            875713, 5105039,
            'data/web-Google_degree.txt',
            'data/web-Google_type.txt',
            0.85,
            0.1,
            0.05,
        ],
    },

    "astroph": {
        "graph_args": [
            18772, 396160,
            'data/ca-AstroPh_degree.txt',
            'data/ca-AstroPh_type.txt',
            0.85,
            0.1,
            0.05,
        ],
    },

    "hepph": {
        "graph_args": [
            12008, 237010,
            'data/ca-HepPh_degree.txt',
            'data/ca-HepPh_type.txt',
            0.85,
            0.1,
            0.05,
        ],
    },
}