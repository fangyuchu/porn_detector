import numpy as np
import re
# a=np.ones((2,5,5,3),dtype=np.float32)
# b=np.array(a)
# b[0,0,0,1]=4
# b[1,2,3,1]=5
# c=np.reshape(b[:,:,:,1],[-1])
# print(b)
# print(np.mean(b,axis=(1,2,0)))
s='vgg1863_bn'
print(re.search(r'(\d+)',s))
