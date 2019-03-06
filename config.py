import numpy as np

#training params
num_epochs=10                       #times for the use of all training data
batch_size=64                       #number of images for one batch
learning_rate=0.005
learning_rate_decay_factor=0.94     #decay factor for exponential decay
weight_decay=5e-4                   # weight decay (L2 penalty)
num_epochs_per_decay=2.5
dropout_rate=0.5
momentum=0.9

#dataset processing params
num_workers=8


#dataset params
#imagenet
imagenet=dict()
imagenet['num_class']=1001                                          #number of the classes
imagenet['label_offset']=1                                          #offset of the label
imagenet['mean']=[0.485, 0.456, 0.406]
imagenet['std']=[0.229, 0.224, 0.225]
imagenet['train_set_size']=1271167
imagenet['validation_set_size']=50000
imagenet['train_set_path']='/home/victorfang/Desktop/imagenet所有数据/imagenet_train'
imagenet['validation_set_path']='/home/victorfang/Desktop/imagenet所有数据/imagenet_validation'

#nsfw
nsfw=dict()
nsfw['num_class']=5
nsfw['mean']=[0.5883667 , 0.51908684, 0.48996434]
            #[0.58538795, 0.5177305 , 0.48874032]
            #[0.5833756 , 0.51693785, 0.48777002]
nsfw['std']=[0.3089234 , 0.3040929 , 0.30784294]
            #[0.31215656, 0.30444726, 0.30714267]
            #[0.31288257, 0.30512026, 0.30701375]
nsfw['train_set_size']=45000
nsfw['validation_set_size']=500
nsfw['train_set_path']='/home/victorfang/Desktop/nsfw_dataset/train'
nsfw['validation_set_path']='/home/victorfang/Desktop/nsfw_dataset/validation'

#model saving params
#how often to write summary and checkpoint
checkpoint_step=4000

# Path for tf.summary.FileWriter and to store model checkpoints
root_path='/home/victorfang/Desktop/pytorch_'
checkpoint_path = "_model_saved/checkpoints"
highest_accuracy_path='_model_saved/accuracy.txt'
global_step_path='_model_saved/global_step.txt'
epoch_path='_model_saved/epoch.txt'