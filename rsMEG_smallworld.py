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

# Small worldness without threshold
# Convert to network
alpha_net = net.network(alpha_fc)
beta_net = net.network(beta_fc)
delta_net = net.network(delta_fc)
gamma_net = net.network(gamma_fc)
theta_net = net.network(theta_fc)

# Compute small world
alpha_small = alpha_net.small_worldness(nrandnet=1, niter=1, normalize=False)
beta_small = beta_net.small_worldness(nrandnet=1, niter=1, normalize=False)
delta_small = delta_net.small_worldness(nrandnet=1, niter=1, normalize=False)
gamma_small = gamma_net.small_worldness(nrandnet=1, niter=1, normalize=False)
theta_small = theta_net.small_worldness(nrandnet=1, niter=1, normalize=False)

plt.figure()
plt.bar(['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta'], [alpha_small, beta_small, delta_small, gamma_small, theta_small])
plt.ylabel('Sigma')
plt.title('Small-World Property')
plt.savefig('Small-World-No-Threshold')
#


# Small worldness over different thresholds
# Take the absolute value
alpha_fc = np.abs(alpha_fc)
beta_fc = np.abs(beta_fc)
delta_fc = np.abs(delta_fc)
gamma_fc = np.abs(gamma_fc)
theta_fc = np.abs(theta_fc)


# Plot degree hist
sns.set()

# Plot smallworld over different thresholds
# Set threshold
small_alpha = []
small_beta = []
small_delta = []
small_gamma = []
small_theta = []

thresholds = np.arange(0,0.31,0.05)
for t in thresholds:

    alpha_fc[alpha_fc < t] = 0
    beta_fc[beta_fc < t] = 0
    delta_fc[delta_fc < t] = 0
    gamma_fc[gamma_fc < t] = 0
    theta_fc[theta_fc < t] = 0

    # Convert to network
    alpha_net = net.network(alpha_fc)
    beta_net = net.network(beta_fc)
    delta_net = net.network(delta_fc)
    gamma_net = net.network(gamma_fc)
    theta_net = net.network(theta_fc)

    # Compute Smallworldness
    small_alpha.append(alpha_net.small_worldness(nrandnet=1, niter=1, normalize=False))
    small_beta.append(beta_net.small_worldness(nrandnet=1, niter=1, normalize=False))
    small_delta.append(delta_net.small_worldness(nrandnet=1, niter=1, normalize=False))
    small_gamma.append(gamma_net.small_worldness(nrandnet=1, niter=1, normalize=False))
    small_theta.append(theta_net.small_worldness(nrandnet=1, niter=1, normalize=False))

# Plot small worldness
sns.set()
plt.figure()
plt.plot(thresholds, small_alpha, label='Alpha')
plt.plot(thresholds, small_beta, label='Beta')
plt.plot(thresholds, small_delta, label='Delta')
plt.plot(thresholds, small_gamma, label='Gamma')
plt.plot(thresholds, small_theta, label='Theta')

plt.title('Small-World Property over Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Small-World Sigma')
plt.legend()
plt.savefig('Small-World.png')

plt.show()


