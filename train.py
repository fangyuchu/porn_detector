import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import resnet
import vgg
import os
import re
from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

def evaluate_model(net,
                   data_loader,
                   save_model,
                   checkpoint_path=None,
                   highest_accuracy_path=None,
                   global_step_path=None,
                   global_step=0,
                   ):
    '''
    :param net: model of NN
    :param data_loader: data loader of test set
    :param save_model: Boolean. Whether or not to save the model.
    :param checkpoint_path: 
    :param highest_accuracy_path: 
    :param global_step_path: 
    :param global_step: global step of the current trained model
    '''
    if save_model:
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise AttributeError('checkpoint path is wrong')
        if highest_accuracy_path is None :
            raise AttributeError('highest_accuracy path is wrong')
        if global_step_path is None :
            raise AttributeError('global_step path is wrong')
        if os.path.exists(highest_accuracy_path):
            f = open(highest_accuracy_path, 'r')
            highest_accuracy = float(f.read())
            f.close()
        else:
            highest_accuracy=0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("{} Start Evaluation".format(datetime.now()))
    print("{} global step = {}".format(datetime.now(), global_step))
    with torch.no_grad():
        correct = 0
        total = 0
        for val_data in data_loader:
            net.eval()
            images, labels = val_data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        correct = float(correct.cpu().numpy().tolist())
        accuracy = correct / total
        print("{} Accuracy = {:.4f}".format(datetime.now(), accuracy))
        if save_model and accuracy > highest_accuracy:
            highest_accuracy = accuracy
            # save model
            print("{} Saving model...".format(datetime.now()))
            torch.save(net.state_dict(), '%s/global_step=%d.pth' % (checkpoint_path, global_step))
            print("{} Model saved ".format(datetime.now()))
            # save highest accuracy
            f = open(highest_accuracy_path, 'w')
            f.write(str(highest_accuracy))
            f.close()
            # save global step
            f = open(global_step_path, 'w')
            f.write(str(global_step))
            print("{} model saved at global step = {}".format(datetime.now(), global_step))
            f.close()

def train(
                    model_name,
                    pretrained=False,
                    dataset_name='nsfw',
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    checkpoint_step=conf.checkpoint_step,
                    checkpoint_path=None,
                    highest_accuracy_path=None,
                    global_step_path=None,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers,
                  ):
    #implemented according to "Pruning Filters For Efficient ConvNets" by Hao Li
    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('cpu')

    temp=re.search(r'(\d+)',model_name).span()[0]
    model=model_name[:temp]                                                     #name of the model.ex: vgg,resnet...
    del temp
    #define the model
    net=getattr(globals()[model],model_name)(pretrained=pretrained).to(device)

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=learning_rate
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    #prepare the data
    if dataset_name is 'nsfw':
        mean=conf.nsfw['mean']
        std=conf.nsfw['std']
        train_set_path=conf.nsfw['train_set_path']
        train_set_size=conf.nsfw['train_set_size']
        validation_set_path=conf.nsfw['validation_set_path']
    # Data loading code
    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std),
    ])
    train = datasets.ImageFolder(train_set_path, transform)
    val = datasets.ImageFolder(validation_set_path, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if checkpoint_path is None:
        checkpoint_path=conf.root_path+model_name+'_porn_detector/checkpoint'
    if highest_accuracy_path is None:
        highest_accuracy_path=conf.root_path+model_name+'_porn_detector/highest_accuracy.txt'
    if global_step_path is None:
        global_step_path=conf.root_path+model_name+'_porn_detector/global_step.txt'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path,exist_ok=True)

    if  os.path.exists(highest_accuracy_path):
        f = open(highest_accuracy_path, 'r')
        highest_accuracy = float(f.read())
        f.close()
        print('highest accuracy from previous training is %f' % highest_accuracy)
        del highest_accuracy

    global_step=0
    if os.path.exists(global_step_path):
        f = open(global_step_path, 'r')
        global_step = int(f.read())
        f.close()
        print('global_step at present is %d' % global_step)
        model_saved_at=checkpoint_path+'/global_step='+str(global_step)+'.pth'
        print('load model from'+model_saved_at)
        net.load_state_dict(torch.load(model_saved_at))
    else:
        print('{} test the model '.format(datetime.now()))                      #no previous checkpoint
        #evaluate_model(net,validation_loader,save_model=False)
    print("{} Start training ".format(datetime.now())+model_name+"...")
    for epoch in range(math.floor(global_step*batch_size/train_set_size),num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()
        # one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            if global_step / math.ceil(train_set_size / batch_size)==epoch+1:               #one epoch of training finished
                evaluate_model(net,validation_loader,
                               save_model=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               global_step_path=global_step_path,
                               global_step=global_step)
                break

            # 准备数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            if step % checkpoint_step == 0 and step != 0:
                evaluate_model(net,validation_loader,
                               save_model=True,
                               checkpoint_path=checkpoint_path,
                               highest_accuracy_path=highest_accuracy_path,
                               global_step_path=global_step_path,
                               global_step=global_step)
                print('{} continue training'.format(datetime.now()))


if __name__ == "__main__":
    train(model_name='vgg16_bn',pretrained=False,checkpoint_step=100,num_epochs=20,batch_size=16)