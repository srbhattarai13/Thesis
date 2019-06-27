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


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:, :, 0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2 ))

    return flow.astype(np.float32)


def normalizedData(inTensor):
    inTensor[inTensor > 20.0 ] = 20.0
    inTensor[ inTensor < -20.0] = -20.0
    inTensor = torch.div(inTensor, 20.0)
    return inTensor

class Datainput(Dataset):

    def __init__(self, df, transform = None):

        self.datafile = pd.read_csv(df)
        self.transform = transform
        self.transformation = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.485,0.456),(0.229,0.224))]


#transforms.Resize((256,256)), transforms.ToTensor(),

    def __len__(self):
        return len(self.datafile)


    def __getitem__(self, idx):
        '''
        :param idx: pass the index of files
        :return: sample generated randomly
        '''

        # Reading Previous Optical flow
        prvFlow1 = self.datafile.iloc[idx,0]
        prvFlow1 = readFlow(prvFlow1)
        prvFlow1 = torch.from_numpy(prvFlow1)
        prvFlow1 = prvFlow1.permute(2,0,1)

        # prvFlow2 = self.datafile.iloc[idx, 1]
        # prvFlow2 = readFlow(prvFlow2)
        #
        # prvFlow3 = self.datafile.iloc[idx, 2]
        # prvFlow3 = readFlow(prvFlow3)
        #
        # # for 4 previous frame
        # prvFlow4 = self.datafile.iloc[idx,3]
        # prvFlow4 = readFlow(prvFlow4)
        #
        # secFlow = np.concatenate((prvFlow1, prvFlow2), axis=2)
        # secFlow2 = np.concatenate((prvFlow3, prvFlow4), axis=2)
        #
        # finalFlow = np.concatenate((secFlow, secFlow2), axis=2)
        #
        # # inFlow = self.transformation(finalFlow)
        # inFlow = torch.from_numpy(finalFlow)
        # inFlow = inFlow.permute(2,0,1)
        # # inFlow = normalizedData(inFlow)
        # inFlow = torch.tanh(inFlow)
        # inFlow = torch.autograd.Variable(inFlow) #torch.from_numpy(prvFlow)
        # inFlow = inFlow.cuda()
        #
        # # Reading Next optical flow
        # nextFlow = self.datafile.iloc[idx,4]
        # nextFlow = readFlow(nextFlow)
        # nextFlow = torch.from_numpy(nextFlow)
        # nextFlow = nextFlow.permute(2,0,1)
        # # nextFlow = normalizedData(nextFlow)
        # nextFlow = torch.tanh(nextFlow)
        # nextFlow = torch.autograd.Variable(nextFlow) #torch.from_numpy(nextFlow)
        # nextFlow = nextFlow.cuda()
        #
        # # reading video length
        # vidLength = self.datafile.iloc[idx, 5]
        # vid_name = self.datafile.iloc[idx, 6]



        # sample = {'inFlow': inFlow, 'nextFlow': nextFlow, 'vidLength':vidLength, 'vidName':vid_name}



        return prvFlow1

#
# # #
# data_opt_img = Datainput('data/allframes.csv')
# print(len(data_opt_img))
#
#
# # Finding out the minimum and maximum value in the Optical flow tensor
# min = 0
# max = 0
# for data in range(0, len(data_opt_img)):
#     inflow = data_opt_img[data]['inFlow']
#     print("Data Frame :{} Min:{} Max: {} ".format(data, torch.min(inflow).item(), torch.max(inflow).item()))
#
#     # if max < torch.max(inflow).item():
    #     max = torch.max(inflow).item()
    # if min > torch.min(inflow).item():
    #     min = torch.min(inflow).item()

    # break
#
#
#     # print(normValue)
# # print(' Min value', min)
# # print(' Max vavlue', max)
# # print('##################################')
# # #
#     if data == 100:
#         break
# train_loader = torch.utils.data.DataLoader(data_opt_img,batch_size=10, shuffle=True)
#
# mean = 0.
# std = 0.
#
# nb_samples = 0.
#
# for data in train_loader:
#     batch_samples = data['inFlow'].size()
#     print(batch_samples)
#
#     break
