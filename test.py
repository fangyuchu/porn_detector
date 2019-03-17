import numpy as np
import re
import os
import random
from PIL import Image
import torch

a=np.array([[1,2],[3,4]])

b=torch.from_numpy(a)

c=b.reshape([1,2,2])
print(c)


