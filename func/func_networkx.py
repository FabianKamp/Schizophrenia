import networkx as nx
import numpy as np
#%% This script contains useful functions that ease the work with network x
def pathlengths(FC):
    """
    Takes Functional Connectivity Matrix and returns the shortest path length between all nodes as array
    :param FC: NxN dimensional array of the functional connectivity
    :return: NxN dimensional array of shortest pathlengths
    """
    N = FC.shape[0]
    G = nx.from_numpy_matrix(FC)
    shortest_path = list(nx.shortest_path_length(G, weight='weight'))
    shortest_path_matrix=np.zeros((N,N))
    for n, tup in enumerate(shortest_path):             # Shortestpath is list of tuples one for each region n
        for m, val in tup[1].items():                   # Element [1] in the tuple contain dictionary of pathlength of n to other regions
            shortest_path_matrix[n,m] = val             # Extract value from dictionary and puts it into matrix
    return shortest_path_matrix