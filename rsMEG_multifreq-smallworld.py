import numpy as np
import func.network as net
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Import the data
# Set directories
parent_dir = 'C:\\Users\\Kamp\\Documents\\SCAN\\Thesis\\repos\\Schizophrenia\\'
subj = 'S126'
results_dir = os.path.join(parent_dir, 'rsMEG', 'Results', subj)
save_dir = os.path.join(parent_dir, 'plots', subj, 'Small-Worldness')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# Load .npy filees of all frequencies
loadFreq = np.array([2,6,10,20,40]) # frequencies to load

dict_sig = {} # Create dictionary that holds all signals
for fcarrier in loadFreq: # Load all signals in one dictionary
    file = os.path.join(results_dir, f'FrqCarrier-{fcarrier}_low-ampl-env_FC.npy')
    fc = np.load(file)
    dict_sig[fcarrier] = {'fc':fc} # Create a subdictionary for the signal

startfreq, endfreq = sorted(dict_sig.keys())[0], sorted(dict_sig.keys())[-1]

# Iteration over all signal in the dictionary
for fcarrier, sub_dict in dict_sig.items():
    # Set diagonal to zero
    np.fill_diagonal(sub_dict['fc'], 0)
    # Convert to network
    signal_net = net.network(sub_dict['fc'])
    sub_dict['smallworld'] = signal_net.small_worldness(nrandnet=1, niter=1, normalize=False)


fig = plt.figure()
smallworld = [sub_dict['smallworld'] for sub_dict in dict_sig.values()]
plt.bar(list(map(lambda x: str(x) + ' Hz', dict_sig.keys())), smallworld)
plt.ylabel('Sigma')
plt.title('Small-World Property')
plt.savefig(os.path.join(save_dir, f'Small-world index ({startfreq}Hz - {endfreq}Hz)'))

# Small worldness over different thresholds
# Take the absolute value
for sub_dict in dict_sig.values():
    sub_dict['abs_fc'] = np.abs(sub_dict['fc'])

# Plot smallworld over different thresholds
# Set threshold
thresholds = np.arange(0,0.31,0.05)

for sub_dict in dict_sig.values():
    sub_dict['abs_smallworld'] = []
    for t in thresholds:
        idx = sub_dict['abs_fc'] < t
        sub_dict['abs_fc'][sub_dict['abs_fc'] < t] = 0

        # Convert to network
        abs_net = net.network(sub_dict['abs_fc'])

        # Compute Smallworldness
        sub_dict['abs_smallworld'].append(abs_net.small_worldness(nrandnet=1, niter=1, normalize=False))

# Plot small worldness
fig = plt.figure()
for fcarrier, sub_dict in dict_sig.items():
    plt.plot(thresholds, sub_dict['abs_smallworld'], label=str(fcarrier)+' Hz')

plt.title('Small-World Property over Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Small-World Sigma')
plt.legend()
plt.savefig(os.path.join(save_dir, f'Small-world index with thresholds ({startfreq}Hz - {endfreq}Hz)'))
plt.show()



