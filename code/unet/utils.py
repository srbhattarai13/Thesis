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


'''Just trying to normalized the data 
def normalizedData(inTensor):
    mean = torch.tensor([0.0484, 0.0043])
    std  = torch.tensor([0.4076, 0.0948])

    inTensor  = (inTensor - mean) / std

    # # simple division Normalization
    # inTensor[inTensor > 20.0 ] = 20.0
    # inTensor[ inTensor < -20.0] = -20.0
    # inTensor = torch.div(inTensor, 20.0)
    return inTensor

'''


def Normalize(tensor):
    mean = (0.0484, 0.0043)
    std = (0.4076, 0.0948)

    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)

    return tensor


def np_load_frame(filename, resize_height=256, resize_width=256):
    img = Image.open(filename)
    img_resized = img.resize((resize_width, resize_height),Image.ANTIALIAS)
    img_numpy = np.array(img_resized)
    img_numpy = img_numpy.astype(dtype=np.float32)
    img_numpy = (img_numpy / 127.5) - 1.0
    inImg = torch.from_numpy(img_numpy)

    return inImg.permute(2,0,1)


class Datainput(Dataset):

    def __init__(self, df):

        self.datafile = pd.read_csv(df)

    def __len__(self):
        return len(self.datafile)


    def __getitem__(self, idx):
        '''
        :param idx: pass the index of files
        :return: sample generated randomly
        '''

        # Reading Previous Optical flow
        frame1_path = self.datafile.iloc[idx,0]
        frame1 = np_load_frame(frame1_path)

        frame2_path = self.datafile.iloc[idx, 1]
        frame2 = np_load_frame(frame2_path)

        frame3_path = self.datafile.iloc[idx, 2]
        frame3 = np_load_frame(frame3_path)

        # for 4 previous frame
        frame4_path = self.datafile.iloc[idx,3]
        frame4  = np_load_frame(frame4_path)

        frame12 = torch.cat((frame1, frame2), 0)
        frame34 = torch.cat((frame3, frame4), 0)
        inputFrame = torch.cat((frame12,frame34),0)

        # target optical flow
        flow_path = self.datafile.iloc[idx,4]
        targetFlow = readFlow(flow_path)
        # targetFlow = Normalize(torch.from_numpy(targetFlow))
        targetFlow_tensor = torch.from_numpy(targetFlow)
        targetFlow_arranged = targetFlow_tensor.permute(2,0,1)


        sample = {'inputFrame': inputFrame.cuda(), 'targetFlow': targetFlow_arranged.cuda()}

        return sample

#
# # # # #
# data_opt_img = Datainput('data/training_avenue.csv')
# print(len(data_opt_img))
#
# train_loader = torch.utils.data.DataLoader(data_opt_img,batch_size=1, shuffle=True)
#
# for k, batch in enumerate(train_loader):
#     print(batch['inputFrame'].size(), batch['targetFlow'].size())
#     print(batch['inputFrame'].max(), batch['inputFrame'].min())
#     print(batch['targetFlow'].max(), batch['targetFlow'].min())
#
#     if k == 0:
#         break

    # if k ==10:
    #     break
    #





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
#     #     max = torch.max(inflow).item()
#     # if min > torch.min(inflow).item():
#     #     min = torch.min(inflow).item()
#
#     # break
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
