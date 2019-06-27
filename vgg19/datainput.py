import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from scipy.misc import imread
from scipy.misc import imresize
from math import ceil
from torch.autograd import Variable

from PIL import Image
import torchvision.transforms as transforms


rng = np.random.RandomState(2017)

def Normalize(tensor):

   ## Mean & std of Avenue Dataset
    mean = (0.0484, 0.0043)
    std = (0.4076, 0.0948)

    # Mean & std of ShanghaiTech
    # mean = (0.0057, 0.0052)
    # std = (0.0790, 0.0522)

    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)

    return tensor


def deNormalize(tensor):

    # Mean & std of Avenue Dataset
    # mean = (0.0484, 0.0043)
    # std = (0.4076, 0.0948)

    # Mean & std of ShanghaiTech
    mean = (0.0484, 0.0043)
    std = (0.4076, 0.0948)

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    return tensor




class Datainput(Dataset):

    def __init__(self, df, transform=None):

        self.datafile = pd.read_csv(df)
        self.img_loader = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.datafile)


    def __getitem__(self, idx):
        '''
        :param idx: pass the index of filesall
        :return: sample generated randomly
        '''
        # vgg19 net Model
        frame_path = self.datafile.iloc[idx,0]
        feats_path = self.datafile.iloc[idx, 1]

        frame = Image.open(frame_path)
        image = self.img_loader(frame).float()
        image = Variable(image)
        # image = image.unsqueeze(0)
        image_tensor= image.cuda()

        feats_tensor = torch.from_numpy(np.load(feats_path))


        # targetFlow_tensor = Normalize(targetFlow_tensor)
        sample = {'inputFrame': image_tensor, 'targetFeats': feats_tensor.cuda().squeeze()}

        return sample

