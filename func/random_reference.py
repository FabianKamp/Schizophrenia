import pandas as pd
import numpy as np
import func.network as net
import func.corr_functions as func

def hqs(tc):
    """
    Returns a random network that is matched to the input networks covariance matrix.
    Using Hirschberger-Qi-Steuer Algorithm as cited in Zalesky 2012b

    :return: n x n dimensional pd.Dataframe
    TODO debug code, control if input has to be is positive finite
    """
    if not isinstance(tc, (np.ndarray, pd.DataFrame)):
        raise ValueError('Timecourse has to be np.ndarray or pd.DataFrame')
    C = func.covariance_mat(tc)

    diag_sum = np.sum(np.diagonal(C))
    diag_len = len(np.diagonal(C))
    diag_mean = diag_sum / diag_len

    off_mean = (np.sum(C) - diag_sum) / (C.size - diag_len)
    off_var = C - off_mean
    np.fill_diagonal(off_var,0)                     # Sets diagonal values to zero
    off_var = np.sum(off_var ** 2)/(C.size-diag_len)

    m = max(2, (diag_mean ** 2 - (off_mean ** 2) / off_var))
    mu = np.sqrt(off_mean / off_var)
    sigma = (-mu) ** 2 + np.sqrt(mu ** 4 + (off_var / m))
    X = np.random.normal(mu, sigma, C.size).reshape(C.shape)
    random_covmat = np.matmul(X, X.T)
    covmat_diag=np.diagonal(random_covmat)
    inv_covmat_diag=np.diag(1/(covmat_diag))        # Invert the diagonal array and convert into matrix
    random_adjacency_mat=np.matmul((np.matmul(inv_covmat_diag, random_covmat)), inv_covmat_diag)    # Convert from covariance matrix into correlation matrix
    random_net=net.network(random_adjacency_mat)
    return random_net

def rewired_net(adjacency, niter=10, seed=None):
    """
    Generates random reference network using the Maslov-Sneppen Algorithm.
    :param adjacency: n x n dimensional pd.DataFrame
    :param niter: int specifying number of iterations to randomize the network
    :param seed: int seed of randomization to make network reproducible
    :return: network.network random network that can be used as null model
    """
    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Input must be numpy.ndarray or panda.DataFrame."
    assert adjacency.shape[0] == adjacency.shape[1], "Adjacency matrix must be square."
    if seed: np.random.seed(seed)                   # Set seed to make random network replicable

    num_nodes = adjacency.shape[0]        # Specify number of nodes
    if isinstance(adjacency, pd.DataFrame):
        node_list=list(adjacency.index)
        adjacency = np.asarray(adjacency)
    else:
        node_list = np.arange(num_nodes)  # Specify list of nodes for index

    adj_mat = np.array(adjacency, copy=True)        # Create two copies of the adjacency matrix
    random_adj = np.array(adj_mat, copy=True)       # Random network is rewired in the following
    for r in range(niter):                          # Number rewiring iterations
        edge_list = [(i,j) for i in range(num_nodes) for j in range(num_nodes) if j != i and not np.isclose(random_adj[i,j], 0)] # Create list of all edges
        num_edges = len(edge_list)/2  # number of edges
        rewired = []  # create list of rewired edges
        e = 0   # counter for rewired edges
        s = 0   # counter that stops rewiring if there are no matching edge pairs are found during the last 30 iterations

        while e<num_edges and s<30:
            first_idx, second_idx = np.random.choice(len(edge_list), 2, replace=False) # Randomly choose two different edges in network
            i, j = edge_list[first_idx]                                     # Node indices in adjacency matrix of first edge
            n, m = edge_list[second_idx]                                    # Node indices in adjacency matrix of second edge

            if n in [i,j] or m in [i,j]:                        # All nodes should be different
                s += 1
                continue
            if (i,n) in rewired or (j,m) in rewired:            # Check if the new edge was already rewired
                s += 1
                continue
            s = 0

            random_adj[i,n]=adj_mat[i,j]             # Rewire i to n
            random_adj[n,i]=adj_mat[i,j]             # Mirror edge to yield complete adjacency mat
            random_adj[j,m]=adj_mat[n,m]             # Rewire j to m
            random_adj[m,j]=adj_mat[n,m]             # Mirror edge

            edge_list=[edge for edge in edge_list if edge != (i,j) and edge != (j,i) and edge != (n,m) and edge != (m,n)]
            rewired.extend([(i,n),(j,m),(n,i),(m,j)])
            e += 2
        adj_mat = np.array(random_adj, copy=True)

    random_adj=pd.DataFrame(random_adj, index=node_list, columns=node_list) # Convert to DataFrame
    return random_adj

def rewire_nx(adjacency, niter=10, seed=None):
    """
    Takes in adjacency matrix, translates it into a networkX network, generates a random reference network
    :param adjacency: nxn dimensional adjacency matrix
    :param niter: number of rewiring iterations
    :param seed: seed for random number generation
    :return: pd.DataFrame of rewired adjacency matrix
    """
    import networkx as nx

    assert isinstance(adjacency, (pd.DataFrame, np.ndarray)), "Input must be numpy.ndarray or panda.DataFrame."
    if isinstance(adjacency, pd.DataFrame):
        nodes = list(adjacency.index)
    else:
        nodes = np.arange(adjacency.shape[0])
    graph = nx.from_numpy_matrix(np.asarray(adjacency))                 # Convert to networkX graph
    random_nx = nx.random_reference(graph, niter=niter, seed=seed)      # Generate random reference using networkX
    random_adj = random_nx.to_numpy_matrix                              # Convert to numpy
    random_adj = pd.DataFrame(random_adj, columns=nodes, index=nodes)   # Converts to pd.DataFrame

    return random_adj


