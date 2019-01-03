import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


transformation = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])


train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',train=True, download=True, transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081))
                                                ])),
                                            batch_size = 32, shuffle=True
                                        )

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',train=False, transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081))
                                                        ])),
                                          batch_size=32, shuffle=True
                                          )

def plot_img(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image,cmap='gray')
    plt.show()


for i in range(0,2):
    plot_img(train_loader[1][0])

