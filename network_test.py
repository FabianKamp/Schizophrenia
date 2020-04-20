import numpy as np
from func import network as net
from func import func_networkx as fnx
import os
import networkx as nx
import matplotlib.pyplot as plt

#%%
# Script to control the network.py file. Each function of network.py is checked if it delivers the same values as networkX

#Load data
os.chdir('rsMEG/orth_results')
alpha_fc = np.load('Alpha_low-ampl-env_FC.npy')
alpha_fc=np.abs(alpha_fc)

alpha_net = net.network(alpha_fc)
alpha_nx = nx.from_numpy_matrix(alpha_fc)

# Degree
nx_degree  = np.array(list(alpha_nx.degree(weight='weight')))[:,1]
diff_degree = nx_degree - net_degree # Is zero
print(diff_degree)

nx_clust = list(nx.clustering(alpha_nx, weight='weight').values())
net_clust = alpha_net.clust_coeff(normalize=True)['node_cluster']
diff_clust  = nx_clust - net_clust

# Shortest path: ok
nx_shortestpath = fnx.pathlengths(alpha_fc)
net_shortestpath = alpha_net.shortestpath()
diff_shortestpath = nx_shortestpath - net_shortestpath['Path_Length'] # returns all zeros -> shortest path is the same
print(diff_shortestpath) # Diff = 0

# Characteristic path length: ok
nx_char_pathlength = nx.average_shortest_path_length(alpha_nx, weight='weight')
net_char_pathlength = alpha_net.char_path(shortestpath=net_shortestpath['Path_Length'])
diff_charpath = (nx_char_pathlength - net_char_pathlength['characteristic_path'])
print(diff_charpath) # Diff = 0

# Global Efficiency: not defined for weighted graphs in networkX
# Network X compute the global efficiency without weights
nx_glob_eff = nx.global_efficiency(alpha_nx)
net_glob_eff = alpha_net.glob_efficiency(shortestpath=net_shortestpath['Path_Length'])
diff_glob_eff = nx_glob_eff - net_glob_eff
print(diff_glob_eff) # Diff = -17

# Clustering Coefficient: Not ok, gets underestimated by networkX
# The following differences occur in NetworkX during the calculation of the clustering of each node:
# 1. The degrees are not weighted. NetworkX just takes the number of all connections divides by those.
# I tested this by setting the degrees in my network all to 90 which yielded the same values as networkX
# 2. The weights are normalized with the maximal weight in the graph
# 3. The weighted geometric mean of the triangles is missing the division by 2.

nx_clust = list(nx.clustering(alpha_nx, weight='weight').values())
net_clust = alpha_net.clust_coeff(normalize=False)['node_cluster']
diff_clust  = nx_clust - net_clust

nx_clust_coeff = nx.average_clustering(alpha_nx, weight='weight')
net_clust_coeff = alpha_net.clust_coeff()['net_cluster']
diff_clust_coeff = nx_clust_coeff - net_clust_coeff
