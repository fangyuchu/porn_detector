import os
from PIL import Image
from PIL import ImageFile
import numpy as np
import torchvision
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
def image_resize(img, img_size):
    """调整图片大小
    """
    if img.mode is not 'RGB':
        img = img.convert('RGB')
    img = img.resize((img_size,img_size))

    return img

def load_img(dir,img_size):
    #load images from 2 level directory structure
    sub_dirs = os.listdir(dir)
    num_img=0
    for sd in sub_dirs:
        img_paths=os.listdir(dir+'/'+sd)
        num_img+=len(img_paths)
    load_img_num=4000
    if num_img<load_img_num:
        print('Seriously? Only these images?')
        return None
    np_img=np.zeros(shape=(load_img_num,img_size,img_size,3),dtype=np.float32)
    for sd in sub_dirs:
        img_paths=os.listdir(dir+'/'+sd)
        index=0
        random_img_index=random.sample([j for j  in range(len(img_paths))],int(load_img_num/len(sub_dirs)))
        for i in random_img_index: #range(int(load_img_num/len(sub_dirs))):
            img = Image.open(dir+'/'+sd+ '/' + img_paths[i])
            img=image_resize(img,img_size)
            arr = np.asarray(img, dtype="float32")
            np_img[index,:,:,:]=arr
            index+=1
    return np_img

def mean_and_std(np_img):
    #calculate the mean and std of sampled image in nparray form
    np_img=np_img/255
    mean=np.mean(np_img,axis=(0,1,2))
    std=np.std(np_img,axis=(0,1,2))
    return mean,std
    #for sd in sub_dirs:

    # for i in range(len(normal_img)):
    #     im = Image.open(normal_img[i])
    #     im_array = np.array(im)
    #     normal_img_data.append(im_array)

if __name__=='__main__':
    print(mean_and_std(load_img('/home/victorfang/Desktop/nsfw_dataset/train',224)))