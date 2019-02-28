import shutil
import os
import random
def generate_train_set(set_size):
    #generate train set from all dataset
    #number of each set is set_size
    l=['drawings','hentai','porn','sexy','neutral']
    for c in l:
        src_dir=u"/home/victorfang/Desktop/nsfw_dataset/"+c
        obj_dir="/home/victorfang/Desktop/nsfw_dataset/train/"+c

        if os.path.exists(obj_dir):
            ls = os.listdir(obj_dir)
            for i in ls:
                c_path = os.path.join(obj_dir, i)
                os.remove(c_path)
        else:
            os.mkdir(obj_dir)

        src_pic=os.listdir(src_dir)
        src_pic=random.sample(src_pic,set_size)
        for i in src_pic:
            oldname= src_dir+"/"+i
            newname=obj_dir+"/"+i
            shutil.copyfile(oldname,newname)

if __name__=='__main__':
    generate_train_set(9000)