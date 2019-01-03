import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt


rng = np.random.RandomState(2017)

class Datainput(Dataset):

    def __init__(self, df, transform=None):

        self.datafile = pd.read_csv(df)
        self.transform = transform
        self.img_loader = transforms.Compose([transforms.Resize((256,256)),
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

        # read the optical flow of same frame
        flow = self.datafile.iloc[idx, 2]
        flow = np.fromfile(flow,np.int32)
        flow = torch.autograd.Variable(torch.from_numpy(flow))
        flow = flow.squeeze()
        flow = flow.cuda()

        videos = self.datafile.iloc[idx,0]

        sample = {'video': videos, 'frame': image, 'flow': flow}

        # print(sample)

        return sample


# data_frame_feats = Datainput('./data/traingListAVENUE.csv')



# print(len(data_frame_feats))


# for i, sample in enumerate(data_frame_feats):
#     print(i)



# print(data_frame_feats)

#
# print(data_frame_feats)
# train_loader = DataLoader(data_frame_feats,batch_size=10, shuffle=True, num_workers=1)
# imgs, steering_angle = next(iter(train_loader))
# print('Batch shape:', imgs.numpy().shape)
# plt.imshow(imgs.numpy()[0,:,:,:])
# plt.show()
