import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from scipy.misc import imread
import os, math, random

rng = np.random.RandomState(2017)
divisor = 64


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


# def Normalize(tensor):
#
#     ## Mean & std of Avenue Dataset
#     mean = (0.5097, 0.2108)
#     std = (1.8202, 0.9454)
#
#     # Mean & std of ShanghaiTech
#     # mean = (0.0057, 0.0052)
#     # std = (0.0790, 0.0522)
#
#     for t, m, s in zip(tensor, mean, std):
#        t.sub_(m).div_(s)
#
#     return tensor
#
#
# def deNormalize(tensor):
#
#     # Mean & std of Avenue Dataset
#     mean = (0.5097, 0.2108)
#     std = (1.8202, 0.9454)
#
#     # Mean & std of ShanghaiTech
#     # mean = (0.0484, 0.0043)
#     # std = (0.4076, 0.0948)
#
#     for t, m, s in zip(tensor, mean, std):
#         t.mul_(s).add_(m)
#
#     return tensor


# def np_load_frame(frame1, frame2):
#
#
#     images = [im1_resized, im2_resized]
#
#     images = np.array(images).transpose(3,0,1,2)
#
#     # ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
#     ims_tensor = torch.from_numpy(images.astype(np.float32))
#
#     # ims_normalize = (ims_tensor / 127.5) - 1.0
#
#     ims_v = Variable(ims_tensor.cuda(), requires_grad=True)
#
#     return ims_v.squeeze()



class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1 + self.th), self.w1:(self.w1 + self.tw), :]


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[(self.h - self.th) // 2:(self.h + self.th) // 2, (self.w - self.tw) // 2:(self.w + self.tw) // 2, :]



class Datainput(Dataset):

    def __init__(self, df, is_cropped=True, crop_size=[256, 256], render_size=[-1,-1]):

        self.datafile = pd.read_csv(df)
        self.is_cropped = is_cropped
        self.crop_size = crop_size
        self.render_size = render_size

    def __len__(self):
        return len(self.datafile)


    def __getitem__(self, idx):
        '''
        :param idx: pass the index of filesall
        :return: generate training/testing samples
        '''

        # Reading single frames for calculating Mean & std
        # frame1_path = self.datafile.iloc[idx,0]
        # flw = readFlow(frame1_path) #
        # im_tensor = torch.from_numpy(flw)
        # im_tensor = im_tensor.permute(2,0,1)
        # im_tensor = im_tensor.squeeze()
        # reading two frames

        img1 = imread(self.datafile.iloc[idx,0])
        img2 = imread(self.datafile.iloc[idx, 1])

        frame_size = img1.shape
        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (frame_size[0] % 64) or (frame_size[1] % 64):
            self.render_size[0] = ((frame_size[0]) // 64) * 64
            self.render_size[1] = ((frame_size[1]) // 64) * 64

        # target optical flow
        flow = readFlow(self.datafile.iloc[idx,2])

        images = [img1, img2]
        image_size = img1.shape[:2]
        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3, 0, 1, 2)
        flow = flow.transpose(2, 0, 1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        sample = {'inputFrame': images.cuda(), 'targetFlow': flow.cuda()}

        return sample

# # # # # #
# data_opt_img = Datainput('data/training_shanghaitech.csv', is_cropped=True)
#
# print(len(data_opt_img))
# #
# train_loader = torch.utils.data.DataLoader(data_opt_img,batch_size=1, shuffle=False)
# #
# for k, batch in enumerate(train_loader):
#     print(batch['inputFrame'].size())
#     break
#
#     # print(batch['inputFrame'].size(), batch['targetFlow'].size(), batch['Equal'])
#     # print(batch['inputFrame'].max(), batch['inputFrame'].min())
#     # print(batch['targetFlow'].max(), batch['targetFlow'].min())
#
#
#     break
#
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
