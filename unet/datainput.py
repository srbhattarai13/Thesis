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



rng = np.random.RandomState(2017)


# def readFlow(name):
#     if name.endswith('.pfm') or name.endswith('.PFM'):
#         return readPFM(name)[0][:, :, 0:2]
#
#     f = open(name, 'rb')
#
#     header = f.read(4)
#     if header.decode("utf-8") != 'PIEH':
#         raise Exception('Flow file header does not contain PIEH')
#
#     width = np.fromfile(f, np.int32, 1).squeeze()
#     height = np.fromfile(f, np.int32, 1).squeeze()
#     flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2 ))
#
#     return flow.astype(np.float32)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

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


def np_load_frame(frame1, frame2, resize_height=256, resize_width=256):
    im1 = imread(frame1)
    im2 = imread(frame2)

    im1_resized = imresize(im1,(resize_width, resize_height))
    im2_resized = imresize(im2, (resize_width, resize_height))

    ims = np.array([[im1_resized, im2_resized]]).astype(np.float32)
    ims = np.resize(ims, (6,256,256))

    ims_tensor = torch.from_numpy(ims)

    ims_normalize = (ims_tensor / 127.5) - 1.0

    ims_v = Variable(ims_normalize.cuda(), requires_grad=True)

    return ims_v.squeeze()


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
        frame2_path = self.datafile.iloc[idx, 1]

        inputFrame = np_load_frame(frame1_path, frame2_path)


        # target optical flow
        flow_path = self.datafile.iloc[idx,2]
        targetFlow = readFlow(flow_path)

        # eql = False
        # if(np.array_equal(targetFlow, targetFlow1)):
        #     eql = True
        # targetFlow = Normalize(torch.from_numpy(targetFlow))
        targetFlow_tensor = torch.from_numpy(targetFlow)
        targetFlow_tensor = Normalize(targetFlow_tensor)
        targetFlow_arranged = targetFlow_tensor.permute(2,0,1)


        sample = {'inputFrame': inputFrame, 'targetFlow': targetFlow_arranged.cuda()}

        return sample

#
# # # # # #
# data_opt_img = Datainput('data/training_avenue.csv')
#
# train_loader = torch.utils.data.DataLoader(data_opt_img,batch_size=1, shuffle=True)
#
# for k, batch in enumerate(train_loader):
#     print(batch['inputFrame'].size(), batch['targetFlow'].size())
#     print(batch['inputFrame'].max(), batch['inputFrame'].min())
#     print(batch['targetFlow'].max(), batch['targetFlow'].min())
#
#     break









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
