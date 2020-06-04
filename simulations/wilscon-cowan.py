import scipy.io
from neurolib.models.wc import WCModel
import matplotlib.pyplot as plt
from neurolib.utils import functions
import os
import loadData
import numpy as np

ds = loadData.Dataset(Group="SCZ")
print(np.max(ds.Cmat))

Dmat = ds.LengthMat
Cmat = ds.Cmat

wc = WCModel(Cmat=Cmat, Dmat=Dmat)
wc.params['exc_ext'] = [0.65] * wc.params['N']
wc.params['signalV'] = 0
wc.params['duration'] = 2 * 1000
wc.params['sigma_ou'] = 0.14
wc.params['K_gl'] = 3.15

wc.run(chunkwise=True)
plt.figure()
for n in range(wc.exc.shape[0]):
    plt.plot(wc.outputs.t, wc.exc[n,:] + n)
fc = functions.fc(wc.exc[:,-10000:])

plt.figure()
plt.imshow(fc)
plt.show()


