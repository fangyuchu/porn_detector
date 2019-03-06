import numpy as np
import re
import os
import random
from PIL import Image
# a=np.ones((2,5,5,3),dtype=np.float32)
# b=np.array(a)
# b[0,0,0,1]=4
# b[1,2,3,1]=5
# c=np.reshape(b[:,:,:,1],[-1])
# print(b)
# print(np.mean(b,axis=(1,2,0)))


# s='vgg1863_bn'
# print(re.search(r'(\d+)',s))

    #sample images from data set whose structure need to be 2 level
dir='/home/victorfang/Desktop/nsfw_dataset/train'
sub_dirs = os.listdir(dir)
num_img=0                                                                       #total number of images in train set
for sd in sub_dirs:
    img_paths=os.listdir(dir+'/'+sd)
    print(sd)
    for p in img_paths:
        img = Image.open(dir + '/' + sd + '/' + p)


