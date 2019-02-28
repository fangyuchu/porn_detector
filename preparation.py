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
    #sample images from data set whose structure need to be 2 level
    sub_dirs = os.listdir(dir)
    num_img=0                                                                       #total number of images in train set
    for sd in sub_dirs:
        img_paths=os.listdir(dir+'/'+sd)
        num_img+=len(img_paths)
    load_img_num=5000                                                                #number of images sampled
    if num_img<load_img_num:
        print('Seriously? Only these images?')
        return None
    np_img=np.zeros(shape=(load_img_num,img_size,img_size,3),dtype=np.float32)
    index = 0

    for sd in sub_dirs:
        img_paths=os.listdir(dir+'/'+sd)
        random_img_index=random.sample([j for j  in range(len(img_paths))],int(load_img_num/len(sub_dirs)))
        temp=0
        for i in random_img_index:
            img = Image.open(dir+'/'+sd+ '/' + img_paths[i])
            img=image_resize(img,img_size)
            arr = np.asarray(img, dtype="float32")
            np_img[index,:,:,:]=arr
            index+=1
            temp+=1
            # print(temp,index)
    return np_img

def mean_and_std(np_img):
    #calculate the mean and std of sampled image in nparray form
    np_img=np_img/255
    R=np_img[:,:,:,0].reshape([-1])
    G=np_img[:,:,:,1].reshape([-1])
    B=np_img[:,:,:,2].reshape([-1])
    np_img=np.array([R,G,B])
    mean=np.mean(np_img,axis=1)
    # temp_mean1=np.mean(np_img,axis=(0,1))
    # temp_mean2=np.mean(np_img,axis=0)
    # temp_mean3=np.mean(temp_mean2,axis=0)
    # temp_mean4=np.mean(temp_mean3,axis=0)
    std=np.std(np_img,axis=1)
    return mean,std


if __name__=='__main__':
    print(mean_and_std(load_img('/home/victorfang/Desktop/nsfw_dataset/train',224)))