import numpy as np
import func.network as net
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data
os.chdir('rsMEG/results144')  # Change directory

alpha_fc = np.load('Alpha_low-ampl-env_FC.npy')
beta_fc = np.load('Beta_low-ampl-env_FC.npy')
delta_fc = np.load('Delta_low-ampl-env_FC.npy')
gamma_fc = np.load('Gamma_low-ampl-env_FC.npy')
theta_fc = np.load('Theta_low-ampl-env_FC.npy')

# Set diagonal to zero
np.fill_diagonal(alpha_fc,0)
np.fill_diagonal(beta_fc,0)
np.fill_diagonal(delta_fc,0)
np.fill_diagonal(gamma_fc,0)
np.fill_diagonal(theta_fc,0)

# Take absolute values
alpha_fc = np.abs(alpha_fc)
beta_fc = np.abs(beta_fc)
delta_fc = np.abs(delta_fc)
gamma_fc = np.abs(gamma_fc)
theta_fc = np.abs(theta_fc)

# Plot histogramm of weights
sns.set()
alpha_edges = [alpha_fc[i,j] for i in range(alpha_fc.shape[0]) for j in range(i+1, alpha_fc.shape[1])]
beta_edges = [beta_fc[i,j] for i in range(beta_fc.shape[0]) for j in range(i+1, beta_fc.shape[1])]
delta_edges = [delta_fc[i,j] for i in range(delta_fc.shape[0]) for j in range(i+1, delta_fc.shape[1])]
gamma_edges = [gamma_fc[i,j] for i in range(gamma_fc.shape[0]) for j in range(i+1, gamma_fc.shape[1])]
theta_edges = [theta_fc[i,j] for i in range(theta_fc.shape[0]) for j in range(i+1, theta_fc.shape[1])]

plt.figure()
plt.hist(alpha_edges, bins=50, alpha=0.5, histtype='stepfilled', label='Alpha')
plt.hist(beta_edges, bins=50, alpha=0.5, histtype='stepfilled', label='Beta')
plt.hist(delta_edges, bins=50, alpha=0.5, histtype='stepfilled', label='Delta')
plt.hist(gamma_edges, bins=50, alpha=0.5, histtype='stepfilled', label='Gamma')
plt.hist(theta_edges, bins=50, alpha=0.5, histtype='stepfilled', label='Theta')

plt.legend(ncol=2)
plt.xlabel('Weight')
plt.ylabel('Count')
plt.title('Weight Distribution')
plt.savefig('Weight Distribution')

# Convert to network
alpha_net = net.network(alpha_fc)
beta_net = net.network(beta_fc)
delta_net = net.network(delta_fc)
gamma_net = net.network(gamma_fc)
theta_net = net.network(theta_fc)

# Plot degree distribution
alpha_degree = alpha_net.degree()
beta_degree  = beta_net.degree()
delta_degree = delta_net.degree()
gamma_degree = gamma_net.degree()
theta_degree = theta_net.degree()

# Plot degree bar plots
plt.figure(figsize=(10,40))
plt.subplot(5,1,1)
plt.bar(np.arange(90), alpha_degree)
plt.ylabel('Alpha')
plt.title('Degrees')
plt.subplot(5,1,2)
plt.bar(np.arange(90), beta_degree)
plt.ylabel('Beta')
plt.subplot(5,1,3)
plt.bar(np.arange(90), delta_degree)
plt.ylabel('Delta')
plt.subplot(5,1,4)
plt.bar(np.arange(90), gamma_degree)
plt.ylabel('Gamma')
plt.subplot(5,1,5)
plt.bar(np.arange(90), theta_degree)
plt.ylabel('Theta')
plt.xlabel('Nodes')
plt.show()

plt.figure()
plt.hist(alpha_degree, bins=10, histtype='stepfilled', alpha=0.5, label='Alpha')
plt.hist(beta_degree, bins=10, histtype='stepfilled', alpha=0.5, label='Beta')
plt.hist(delta_degree, bins=10, histtype='stepfilled', alpha=0.5, label='Delta')
plt.hist(gamma_degree, bins=10, histtype='stepfilled', alpha=0.5, label='Gamma')
plt.hist(theta_degree, bins=10, histtype='stepfilled', alpha=0.5, label='Theta')

plt.legend(ncol=2, loc='upper left')
plt.title('Degree Distribution')
plt.ylabel('Count')
plt.xlabel('Degree')
plt.savefig('Degree_Distribution')

# Compute shortest path
alpha_shrt = alpha_net.shortestpath()
beta_shrt = beta_net.shortestpath()
delta_shrt = delta_net.shortestpath()
gamma_shrt = gamma_net.shortestpath()
theta_shrt = theta_net.shortestpath()

# Plot Shortest Path and Functional Connectivity
sns.reset_defaults()
# Alpha
plt.figure(figsize=(10,8))
plt.subplot(2,2,1); plt.imshow(alpha_fc)
plt.colorbar(); plt.title('Functional Connectivity')
plt.ylabel('Alpha')
plt.subplot(2,2,2); plt.imshow(alpha_shrt)
plt.colorbar(); plt.title('Shortest Path')
# Beta
plt.subplot(2,2,3); plt.imshow(beta_fc)
plt.colorbar(); plt.ylabel('Beta')
plt.subplot(2,2,4); plt.imshow(beta_shrt)
plt.colorbar()
plt.savefig('Alpha-Beta_ShortestPath')
# Delta
plt.figure(figsize=(10,8))
plt.subplot(2,2,1); plt.imshow(delta_fc)
plt.colorbar(); plt.ylabel('Delta')
plt.title('Functional Connectivity')
plt.subplot(2,2,2); plt.imshow(delta_shrt)
plt.colorbar(); plt.title('Shortest Path')
# Gamma
plt.subplot(2,2,3); plt.imshow(gamma_fc)
plt.colorbar(); plt.ylabel('Gamma')
plt.subplot(2,2,4); plt.imshow(gamma_shrt)
plt.colorbar()
plt.savefig('Delta-Gamma_ShortestPath')
# Theta
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(theta_fc)
plt.colorbar(); plt.ylabel('Theta')
plt.title('Functional Connectivity')
plt.subplot(1,2,2); plt.imshow(theta_shrt)
plt.colorbar(); plt.title('Shortest Path')
plt.savefig('Theta_ShortestPath')

# Plot histogramm of shortest path
# Convert path matrix to list
alpha_shrt = [alpha_shrt.iloc[i,j] for i in range(alpha_fc.shape[0]) for j in range(i+1, alpha_fc.shape[1])]
beta_shrt = [beta_shrt.iloc[i,j] for i in range(beta_fc.shape[0]) for j in range(i+1, beta_fc.shape[1])]
delta_shrt = [delta_shrt.iloc[i,j] for i in range(delta_fc.shape[0]) for j in range(i+1, delta_fc.shape[1])]
gamma_shrt = [gamma_shrt.iloc[i,j] for i in range(gamma_fc.shape[0]) for j in range(i+1, gamma_fc.shape[1])]
theta_shrt = [theta_shrt.iloc[i,j] for i in range(theta_fc.shape[0]) for j in range(i+1, theta_fc.shape[1])]

sns.set()
plt.figure()
plt.hist(alpha_shrt, bins=50, alpha=0.5, histtype='stepfilled', label='Alpha')
plt.hist(beta_shrt, bins=50, alpha=0.5, histtype='stepfilled', label='Beta')
plt.hist(delta_shrt, bins=50, alpha=0.5, histtype='stepfilled', label='Delta')
plt.hist(gamma_shrt, bins=50, alpha=0.5, histtype='stepfilled', label='Gamma')
plt.hist(theta_shrt, bins=50, alpha=0.5, histtype='stepfilled', label='Theta')
plt.legend(ncol=2)
plt.title('Distribution of Shortest Path Lengths')
plt.xlabel('Shortest Path Length')
plt.ylabel('Count')
plt.savefig('Shortest-Path_Distribution')

# Plot histogramm  of characteristic path lengths
alpha_char = alpha_net.char_path(node_by_node=True)
beta_char  = beta_net.char_path(node_by_node=True)
delta_char = delta_net.char_path(node_by_node=True)
gamma_char = gamma_net.char_path(node_by_node=True)
theta_char = theta_net.char_path(node_by_node=True)

plt.figure()
plt.hist(alpha_char, bins=15, histtype='stepfilled', alpha=0.5, label='Alpha')
plt.hist(beta_char, bins=15, histtype='stepfilled', alpha=0.5, label='Beta')
plt.hist(delta_char, bins=15, histtype='stepfilled', alpha=0.5, label='Delta')
plt.hist(gamma_char, bins=15, histtype='stepfilled', alpha=0.5, label='Gamma')
plt.hist(theta_char, bins=15, histtype='stepfilled', alpha=0.5, label='Theta')

plt.legend(ncol=2)
plt.title('Characteristic Path Length Distribution')
plt.ylabel('Count')
plt.xlabel('Char. Path Length')
plt.savefig('Char-Path_Distribution')

# Plot Avg neigh degree
alpha_avg_neigh = alpha_net.avg_neigh_degree()
beta_avg_neigh = beta_net.avg_neigh_degree()
delta_avg_neigh = delta_net.avg_neigh_degree()
gamma_avg_neigh = gamma_net.avg_neigh_degree()
theta_avg_neigh = theta_net.avg_neigh_degree()

plt.figure()
plt.hist(alpha_avg_neigh, histtype='stepfilled', alpha=0.5, label='Alpha')
plt.hist(beta_avg_neigh, histtype='stepfilled', alpha=0.5, label='Beta')
plt.hist(delta_avg_neigh, histtype='stepfilled', alpha=0.5, label='Delta')
plt.hist(gamma_avg_neigh, bins=50,histtype='stepfilled', alpha=0.5, label='Gamma')
plt.hist(theta_avg_neigh, histtype='stepfilled', alpha=0.5, label='Theta')

plt.legend(ncol=2, loc='upper right')
plt.title('Avg. Neigh. Degree Distribution')
plt.ylabel('Count')
plt.xlabel('Avg. Neigh. Degree')
plt.savefig('Avg-Neigh-Degree_Distribution')

# Plot Assortativity
alpha_ass = alpha_net.assortativity()
beta_ass = beta_net.assortativity()
delta_ass = delta_net.assortativity()
gamma_ass = gamma_net.assortativity()
theta_ass = theta_net.assortativity()

plt.figure()
plt.bar(['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta'], [alpha_ass, beta_ass, delta_ass, gamma_ass, theta_ass], color='steelblue')
plt.title('Assortativity')
plt.ylabel('Sigma')
plt.savefig('Assortativity')
plt.show()
