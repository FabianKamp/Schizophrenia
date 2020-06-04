import loadData
import numpy as np
ds = loadData.Dataset(Group='HC')
print(np.diag(ds.Cmat))
