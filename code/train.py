import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from utils import Datainput


# def downcon(input_channel, output_channel, kernel_size=3, stride=1, padding=1):
#     conv1 =


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.downconv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.downconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.drelu1    = nn.ReLU(inplace=True)
        self.dmaxp1    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.downconv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.drelu2    = nn.ReLU(inplace=True)
        self.dmaxp2    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.downconv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.drelu3    = nn.ReLU(inplace=True)
        self.dmaxp3    = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.downconv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.drelu4    = nn.ReLU(inplace=True)


        # upsample

        self.upconv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.urelu1  = nn.ReLU(inplace=True)

        self.upconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.Conv2d(128, 128,  kernel_size=3, stride=1, padding=1)
        self.urelu2  = nn.ReLU(inplace=True)

        self.upconv5 = nn.Conv2d(128,64, kernel_size=3, stride=1, padding=1)
        self.upconv6 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.urelu3  = nn.ReLU(inplace=True)


        # self.up = nn.functional.interpolate(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.downconv1(x)
        x = self.downconv2(x)
        x1 = self.drelu1(x)
        x = self.dmaxp1(x1)


        x = self.downconv3(x)
        x = self.downconv4(x)
        x2 = self.drelu2(x)
        x = self.dmaxp2(x2)

        x = self.downconv5(x)
        x = self.downconv6(x)
        x3 = self.drelu3(x)
        x = self.dmaxp3(x3)

        x = self.downconv7(x)
        x4 = self.downconv8(x)
        x = self.drelu4(x4)

        x = nn.functional.interpolate(x,scale_factor=2, mode='bilinear')
        # x = self.up(x4)

        # x = torch.cat((x,x3), dim=1)
        # x = x.view(-1, 512,64,64)

        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.urelu1(x)

        x = nn.functional.interpolate(x,scale_factor=2, mode='bilinear')

        # x = self.up(x)

        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.urelu2(x)

        x = nn.functional.interpolate(x,scale_factor=2, mode='bilinear')


        # x = self.up(x)

        x = self.upconv5(x)
        x = self.upconv6(x)
        x = self.urelu3(x)


        return x



# function for loading a frame
def loadGT(frame):
    optflow = np.load(frame)
    optflow_variable = torch.autograd.Variable(torch.from_numpy(optflow))
    return optflow_variable



# transforming image into tensor
img_width,img_height = 256,256
img_loader = transforms.Compose([transforms.Resize((img_width,img_height)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])


# load frame
def image_loader(image_name):
    image = Image.open(image_name)
    # image.show()
    image = img_loader(image).float()
    image = Variable(image)
    image = image.unsqueeze(0)

    return image.cuda()


#data inputs
data_opt_img = Datainput('data/traingListAVENUE.csv')

print(data_opt_img.len())
# train_loader = torch.utils.data.DataLoader(data_opt_img,batch_size=32, shuffle=True)
#
#
# for i, batch in enumerate(train_loader):
#     print(i)



# mdl = Unet().cuda()



























# frame_tensor = image_loader("../datasets/avenue/training/optical_flowmap/01/0000.png")
#
# x = mdl(frame_tensor)
#
# x = x.view(256,256,3)
# x = x.detach().cpu()
# trs = transforms.ToPILImage()
# plt.imshow(trs(x))
# plt.show()

# print(x.size())