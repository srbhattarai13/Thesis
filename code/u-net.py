import torch
from torch.utils.data import Dataset, Dataloader
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.jit import script, trace
from torchvision import datasets, models, transforms





class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init_()

        self.conv1 = nn.conv2d(3, 64, kernel_size =3 , stride=1 , padding = 1)
        self.conv2 = nn.conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2 , padding=0)

        self.conv3 = nn.conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.conv2d(128,256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.conv2d(256, kernel_size=3 , stride=2, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv7 = nn.conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.conv2d(512, 512, kernel_size=3, stride=2, padding=1)

        self.upsample = nn.Up




    def forward(selfself,x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))





