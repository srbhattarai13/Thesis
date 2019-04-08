import torch
import  numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data.dataset as Dataset
import pandas as pd
import matplotlib.pyplot as plt


rng = np.random.RandomState(2017)

class Datainput(Dataset.Dataset):

    def __init__(self, df, transform=None):

        self.datafile = pd.read_csv(df)
        self.transform = transform
        self.img_loader = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])


    def __len__(self):
        return len(self.datafile)


    def __getitem__(self, idx):
        '''
        :param idx: pass the index of files
        :return: sample generated randomly
        '''

        frame = self.datafile.iloc[idx,1]

        # read the frame
        image = Image.open(frame)
        image = self.img_loader(image).float()
        image = Variable(image)
        # image = image.unsqueeze(0)
        image = image.cuda()

        # read the features of same frame
        feats = self.datafile.iloc[idx, 2]
        feats = np.load(feats)
        feats = torch.autograd.Variable(torch.from_numpy(feats))
        feats = feats.squeeze()
        feats = feats.cuda()

        videos = self.datafile.iloc[idx,0]

        sample = {'video': videos, 'frame': image, 'feats': feats}

        return sample


