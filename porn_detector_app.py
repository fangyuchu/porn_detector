import tkinter as tk
from tkinter import  filedialog
import config as conf
import torch
import torchvision.transforms as transforms
import vgg
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES=True

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('cpu')
    net = vgg.vgg16_bn(pretrained=False).to(device)
    model_saved_at = "/home/victorfang/Desktop/pytorch_vgg16_bn_porn_detector/checkpoint/global_step=19215.pth"
    net.load_state_dict(torch.load(model_saved_at))
    return net

def load_image(image_path):
    mean = conf.nsfw['mean']
    std = conf.nsfw['std']
    image = Image.open(image_path)

    resize=transforms.Resize(224)
    crop = transforms.RandomResizedCrop(size=224,scale=(1,1))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=mean, std=std)

    image=resize(image)
    image = crop(image)

    image.show()

    image = to_tensor(image)
    image = normalize(image)
    image = image.reshape([1, 3, 224, 224])

    return image

def app(
        image_path,
        ):
    image_class = ['drawings', 'hentai','neutral', 'porn', 'sexy']

    net=load_model()

    while True:
        root=tk.Tk()
        root.withdraw()
        image_path=filedialog.askopenfilename()

        image=load_image(image_path)

        with torch.no_grad():
            net.eval()

            outputs = net.forward(image)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            print(image_class[predicted])



if __name__ == "__main__":
    app("/home/victorfang/Desktop/test/l7xSda9.jpg")