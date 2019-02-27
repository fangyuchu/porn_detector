import numpy as np
a=np.zeros((5,5,5,3),dtype=np.float32)
print((np.std(a,axis=(0,1,2))+5)/9)