import numpy as np
import func.network as net
import func.from_networkx as fnx
from utils.func import *
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Script to control the network.py file. Each function of network.py is checked if it delivers the same values as networkX
#Load data
os.chdir('rsMEG/orth_results')
alpha_fc = np.load('Alpha_low-ampl-env_FC.npy')
alpha_fc=np.abs(alpha_fc)
np.fill_diagonal(alpha_fc,0)

alpha_net = net.network(alpha_fc)
alpha_net2 = net.network(alpha_fc)
alpha_nx = nx.from_numpy_matrix(alpha_fc)
alpha_net2 = net.network(alpha_fc)

## Test area
t = timer()
t.tic()
nx_smallworld = alpha_net.small_worldness(nrandnet=10, niter=10, nx=True)
t.toc()
t.tic()
net_smallworld = alpha_net.small_worldness(nrandnet=1, niter=10, nx=False)
t.toc()
diff_smallworld = nx_smallworld - net_smallworld
###

# Degree: ok
nx_degree  = np.array(list(alpha_nx.degree(weight='weight')))[:,1]
net_degree = alpha_net.degree()
diff_degree = nx_degree - net_degree # Is zero
print('Difference between Degrees:')
print(diff_degree)

# Shortest path: ok
nx_shortestpath = fnx.shortest_path_length(alpha_fc)
net2_shortestpath = alpha_net.shortestpath(nx=True)
net_shortestpath = alpha_net.shortestpath(nx=False)
diff_shortestpath = nx_shortestpath - net_shortestpath # returns all zeros -> shortest path is the same
diff2_shortestpath = net_shortestpath - net2_shortestpath
print('Difference between Shortest Path Length:')
print(diff_shortestpath) # Diff = 0
print(diff2_shortestpath)

# Characteristic path length: ok
nx_char_pathlength = nx.average_shortest_path_length(alpha_nx, weight='weight')
net_char_pathlength = alpha_net.char_path(nx=False)
net2_char_pathlength = alpha_net.char_path(nx=True)
diff_charpath = (nx_char_pathlength - net_char_pathlength)
diff2_charpath = net_char_pathlength - net2_char_pathlength
print(f'Difference between Characteristic Path Length: {diff_charpath}, {diff2_charpath}')


# Global Efficiency: not defined for weighted graphs in networkX
# Network X compute the global efficiency without weights
nx_glob_eff = nx.global_efficiency(alpha_nx)
net_glob_eff = alpha_net.glob_efficiency(nx=False)
net2_glob_eff = alpha_net.glob_efficiency(nx=True)
diff_glob_eff = nx_glob_eff - net_glob_eff
diff2_glob_eff = net_glob_eff - net2_glob_eff
print(f'Global efficiency difference (networkX without weights): {diff_glob_eff}, {diff2_glob_eff}') # Diff = -17


# Clustering Coefficient: not ok, gets underestimated by networkX.
# The following differences occur in NetworkX during the calculation of the clustering of each node:
# 1. The degrees are not weighted. NetworkX just takes the number of all connections divides by those.
# I tested this by setting the degrees in my network all to 89 which yielded the same values as networkX
# 2. The weights are normalized with the maximal weight in the graph
# 3. The weighted geometric mean of the triangles is missing the division by 2.

nx_clust = list(nx.clustering(alpha_nx, weight='weight').values())
net_clust = alpha_net.clust_coeff(node_by_node=True, normalize=False, nx=False)
net2_clust = alpha_net.clust_coeff(node_by_node=True, normalize=False, nx=True)
diff_clust  = nx_clust - net_clust
diff2_clust = net_clust - net2_clust
print('Node by node cluster coefficient difference:')
print(diff_clust)
print(diff2_clust)

nx_clust_coeff = nx.average_clustering(alpha_nx, weight='weight')
net_clust_coeff = alpha_net.clust_coeff(node_by_node=False, normalize=False, nx=False)
net2_clust_coeff = alpha_net.clust_coeff(node_by_node=False, normalize=False, nx=True)
diff_clust_coeff = nx_clust_coeff - net_clust_coeff
diff2_clust_coeff = net_clust_coeff - net2_clust_coeff
print(f'Cluster coefficinet difference: {diff_clust_coeff}, {diff2_clust_coeff}')

# Transitivity: not defined for weighted graphs in networkX
#net_trans = alpha_net.transitivity()
#print(f'Transitivity: {net_trans}')

# Closeness Centrality: ok
nx_close_cent = list(nx.closeness_centrality(alpha_nx, distance='weight').values())
net_close_cent = alpha_net.closeness_centrality(nx=False)
net2_close_cent = alpha_net.closeness_centrality(nx=True)
diff_close_cent = nx_close_cent - net_close_cent
diff2_close_cent = net_close_cent - net2_close_cent
print('Difference closeness centrality: ')
print(diff_close_cent)
print(diff2_close_cent)

# Betweeness Centrality: ok, but NetworkX values are multiplied by 2 (different normalization)
# https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html
nx_between_cent = list(nx.betweenness_centrality(alpha_nx, weight='weight').values())
net_between_cent = alpha_net.betweenness_centrality()
diff_between_cent = nx_between_cent - net_between_cent
print('Difference Betweenness Centrality: ')
print(diff_between_cent) # Diff = net_between_cent

# Small World: not defined in networkX
# Can't be the same, because the clustering coefficient differs between the implementations
# I tested my clustering coefficient using the random_reference implementation from networkX. It turned out that
# in nx.random_reference two edges are not rewired if a connection in the network already exists. This means
# given we randomly chose the connections (a,b) and (c,d) we should rewire like this (a,c) and (b,d). In network x this
# not done if (a,c) and (b,d) exist in the network. For a fully connected network this means that nx.random_reference
# doesn't randomize at all.

nx_smallworld = alpha_net.small_worldness(nrandnet=1, niter=10, nx=True)
net_smallworld = alpha_net.small_worldness(nrandnet=1, niter=10)
diff_smallworld = nx_smallworld - net_smallworld
