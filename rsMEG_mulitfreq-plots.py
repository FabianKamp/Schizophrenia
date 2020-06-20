import numpy as np
import func.network as net
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data
# Set directories
parent_dir = 'C:\\Users\\Kamp\\Documents\\SCAN\\Thesis\\repos\\Schizophrenia\\'
subj = 'S126'
results_dir = os.path.join(parent_dir, 'rsMEG', 'Results', subj)
save_dir = os.path.join(parent_dir, 'plots', subj)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# Load .npy filees of all frequencies
loadFreq = np.arange(24,32,2) # frequencies to load
dict_sig = {}

for fcarrier in loadFreq: # Load all signals in one dictionary
    file = os.path.join(results_dir, f'FrqCarrier-{fcarrier}_low-ampl-env_FC.npy')
    fc = np.load(file)
    dict_sig[fcarrier] = {'fc':fc} # Create a subdictionary for the signal

startfreq, endfreq = sorted(dict_sig.keys())[0], sorted(dict_sig.keys())[-1]

# Iteration over all signal in the dictionary
for fcarrier, sub_dict in dict_sig.items():
    # Set diagonal to zero
    np.fill_diagonal(sub_dict['fc'], 0)
    signal_fc = np.abs(sub_dict['fc'])
    # Take only upper edges
    #signal_edges = [signal_fc[i,j] for i in range(signal_fc.shape[0]) for j in range(i+1, signal_fc.shape[1])]
    upper_tri = np.triu_indices(signal_fc.shape[0], k=1)
    sub_dict['edges'] = signal_fc[upper_tri]

# Plot histogramm of weights
sns.set()
plt.figure()
for fcarrier, sub_dict in dict_sig.items():
    plt.hist(sub_dict['edges'], bins=30, alpha=0.5, histtype='stepfilled', label=str(fcarrier)+' Hz')
plt.legend(loc='best')
plt.xlabel('Weight')
plt.ylabel('Count')
plt.title(f'Weight Distribution')
#plt.show()
plt.savefig(os.path.join(save_dir,f'Weight Distribution ({startfreq} Hz - {endfreq} Hz)'))

# Convert to network and compute graph measures
for fcarrier, sub_dict in dict_sig.items():
    signal_net = net.network(sub_dict['fc'])
    sub_dict['degree'] = signal_net.degree()
    sub_dict['shortestpath'] = signal_net.shortestpath()
    sub_dict['halfshrtpaths'] = [sub_dict['shortestpath'].iloc[i, j] for i in range(sub_dict['fc'].shape[0]) for j in
                                 range(i + 1, sub_dict['fc'].shape[1])]
    sub_dict['charpath'] = signal_net.char_path(node_by_node=True)
    sub_dict['avg_neigh'] = signal_net.avg_neigh_degree()
    sub_dict['ass'] = signal_net.assortativity()


# Plot degree bar plots
fig = plt.figure(figsize=(10,40))
fig.suptitle('Degrees')
cols = 3
rows = np.ceil(len(dict_sig)/cols).astype('int')
counter = 1
for fcarrier, sub_dict in dict_sig.items():
    ax = fig.add_subplot(rows, cols, counter)
    ax.bar(np.arange(90), sub_dict['degree'])
    ax.set_title(str(fcarrier) + ' Hz')
    counter += 1
#plt.show()

# Plot Degree histogramm
plt.figure()
for fcarrier, sub_dict in dict_sig.items():
    plt.hist(sub_dict['degree'], bins=15, histtype='stepfilled', alpha=0.5, label=str(fcarrier)+' Hz')
plt.legend(loc='best')
plt.title('Degree Distribution')
plt.ylabel('Count')
plt.xlabel('Degree')
#plt.show()
plt.savefig(os.path.join(save_dir, f'Degree_Distribution ({startfreq}Hz - {endfreq}Hz)'))

# Plot Shortest Path and Functional Connectivity
sns.reset_defaults()
cols = 2
rows = len(dict_sig)
counter = 1
fig = plt.figure(figsize=(8, len(dict_sig)*3))
for fcarrier, sub_dict in dict_sig.items():
    ax = fig.add_subplot(rows, cols, counter)
    im = ax.imshow(sub_dict['fc'])
    fig.colorbar(im, fraction=0.046);
    if counter == 1:
        ax.set_title('Funct. Connectivity')
    ax.set_ylabel(str(fcarrier)+' Hz');
    counter += 1
    ax = fig.add_subplot(rows, cols, counter)
    im = ax.imshow(sub_dict['shortestpath'])
    fig.colorbar(im, fraction=0.046)
    if counter == 2:
        ax.set_title('Shortest Path')
    counter += 1
plt.savefig(os.path.join(save_dir, f'FC and Shortest Path ({startfreq}Hz - {endfreq}Hz)'))

# Plot histogramm of shortest path
# Convert path matrix to list
sns.set()
plt.figure()
for fcarrier, sub_dict in dict_sig.items():
    plt.hist(sub_dict['halfshrtpaths'], bins=30, alpha=0.5, histtype='stepfilled', label=str(fcarrier)+' Hz')
plt.legend(loc='best', ncol=len(dict_sig)//3+1)
plt.title('Distribution of Shortest Path Lengths')
plt.xlabel('Shortest Path Length')
plt.ylabel('Count')
#plt.show()
plt.savefig(os.path.join(save_dir, f'Shortest Path Destribution ({startfreq}Hz - {endfreq}Hz)'))

# Plot histogramm  of characteristic path lengths
plt.figure()
for fcarrier, sub_dict in dict_sig.items():
    plt.hist(sub_dict['charpath'], bins=15, histtype='stepfilled', alpha=0.5, label=str(fcarrier)+' Hz')

plt.legend(ncol=len(dict_sig)//3+1, loc='best')
plt.title('Characteristic Path Length Distribution')
plt.ylabel('Count')
plt.xlabel('Char. Path Length')

plt.savefig(os.path.join(save_dir, f'Characteristic Path Distribution ({startfreq}Hz - {endfreq}Hz)'))
#plt.show()

# Plot Avg neigh degree
plt.figure()
for fcarrier, sub_dict in dict_sig.items():
    plt.hist(sub_dict['avg_neigh'], histtype='stepfilled', alpha=0.5, label=str(fcarrier)+' Hz')

plt.legend(ncol=2, loc='best')
plt.title('Avg. Neigh. Degree Distribution')
plt.ylabel('Count')
plt.xlabel('Avg. Neigh. Degree')
#plt.show()
plt.savefig(os.path.join(save_dir, f'Avg-Neigh-Degree_Distribution ({startfreq}Hz - {endfreq}Hz)'))

# Plot Assortativity
plt.figure()
ass = [sub_dict['ass'] for sub_dict in dict_sig.values()] # extract assortativity for each frequency
plt.bar(list(map(lambda x: str(x) + ' Hz' ,dict_sig.keys())), ass, color='steelblue')
plt.title('Assortativity')
plt.ylabel('Sigma')
#plt.show()
plt.savefig(os.path.join(save_dir, f'Assortativity ({startfreq}Hz - {endfreq}Hz)'))
