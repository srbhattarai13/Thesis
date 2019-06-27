import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import time
from PIL import Image
from tensorboardX import SummaryWriter
from datainput import Datainput
import matplotlib.pyplot as plt
from flowsd import FlowNet2SD
from math import ceil

import pickle

# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument('testframes', help='path for testing frames')
parser.add_argument('dataset', help=' Name of dataset')
parser.add_argument("--rgb_max", type=float, default=255.)

args = parser.parse_args()

dataset_dir = args.testframes
dataset = args.dataset


UNKNOWN_FLOW_THRESH = 1e7



def make_color_wheel():
  """
  Generate color wheel according Middlebury color code
  :return: Color wheel
  """
  RY = 15
  YG = 6
  GC = 4
  CB = 11
  BM = 13
  MR = 6

  ncols = RY + YG + GC + CB + BM + MR

  colorwheel = np.zeros([ncols, 3])

  col = 0

  # RY
  colorwheel[0:RY, 0] = 255
  colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
  col += RY

  # YG
  colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
  colorwheel[col:col + YG, 1] = 255
  col += YG

  # GC
  colorwheel[col:col + GC, 1] = 255
  colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
  col += GC

  # CB
  colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
  colorwheel[col:col + CB, 2] = 255
  col += CB

  # BM
  colorwheel[col:col + BM, 2] = 255
  colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
  col += + BM

  # MR
  colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
  colorwheel[col:col + MR, 0] = 255

  return colorwheel

def compute_color(u, v):
  """
  compute optical flow color map
  :param u: optical flow horizontal map
  :param v: optical flow vertical map
  :return: optical flow in color code
  """
  [h, w] = u.shape
  img = np.zeros([h, w, 3])
  nanIdx = np.isnan(u) | np.isnan(v)
  u[nanIdx] = 0
  v[nanIdx] = 0

  colorwheel = make_color_wheel()
  ncols = np.size(colorwheel, 0)

  rad = np.sqrt(u ** 2 + v ** 2)

  a = np.arctan2(-v, -u) / np.pi

  fk = (a + 1) / 2 * (ncols - 1) + 1

  k0 = np.floor(fk).astype(int)

  k1 = k0 + 1
  k1[k1 == ncols + 1] = 1
  f = fk - k0

  for i in range(0, np.size(colorwheel, 1)):
    tmp = colorwheel[:, i]
    col0 = tmp[k0 - 1] / 255
    col1 = tmp[k1 - 1] / 255
    col = (1 - f) * col0 + f * col1

    idx = rad <= 1
    col[idx] = 1 - rad[idx] * (1 - col[idx])
    notidx = np.logical_not(idx)

    col[notidx] *= 0.75
    img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

  return img


def flow_to_image(flow, display=False):
  """
  Convert flow into middlebury color code image
  :param flow: optical flow map
  :return: optical flow image in middlebury color
  """
  u = flow[:, :, 0]
  v = flow[:, :, 1]

  maxu = -999.
  maxv = -999.
  minu = 999.
  minv = 999.

  idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
  u[idxUnknow] = 0
  v[idxUnknow] = 0

  maxu = max(maxu, np.max(u))
  minu = min(minu, np.min(u))

  maxv = max(maxv, np.max(v))
  minv = min(minv, np.min(v))

  rad = np.sqrt(u ** 2 + v ** 2)
  maxrad = max(-1, np.max(rad))

  if display:
    print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

  u = u / (maxrad + np.finfo(float).eps)
  v = v / (maxrad + np.finfo(float).eps)

  img = compute_color(u, v)

  idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
  img[idx] = 0

  return np.uint8(img)

def deNormalize(tensor):
  # Mean & std of ShanghaiTech
  mean = (0.0484, 0.0043)
  std = (0.4076, 0.0948)

  for t, m, s in zip(tensor, mean, std):
    t.mul_(s).add_(m)

  return tensor

# loading the data points
testSet = Datainput(dataset_dir)
test_loader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=False)

mdl = FlowNet2SD(args)
mdl.eval()

adapted_height = 384
adapted_width = 640
scale_height = 0.9375
scale_width = 1.0
# loading model
models_list = ['batch_64_epoch_0_Iteration_2000_1e-05_fnet_nextPict_ShTech.pth.tar']
for model in models_list:

  flownet = torch.load('checkpoints/shanghaitech/' + model)
  flownet.eval()
  # mdl.load_state_dict(torch.load('checkpoints/' + dataset + '/' + model))
  # mdl = torch.load('../flownet2-pytorch/pretrained/' + dataset + '/' + model)
  # pretrained_dict = torch.load('../flownet2-pytorch/pretrained/' + model)['state_dict']
  # mdl.load_state_dict(pretrained_dict)
  #
  # model_dict = mdl.state_dict()
  # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  # model_dict.update(pretrained_dict)
  # mdl.load_state_dict(model_dict)
  # mdl.cuda()
  flownet.cuda()
  for k, test in enumerate(test_loader):
    print('Frame Number: ', k)

    predFlow = flownet(test['inputFrame']).cpu().data
    deNormFlow = predFlow[0].numpy().transpose((1, 2, 0))
    flow_im = flow_to_image(deNormFlow)
    image = Image.fromarray(flow_im, 'RGB')
    image = image.resize((int(scale_width * adapted_width), int(scale_height * adapted_height)), Image.ANTIALIAS)
    image.save('prediction/01/2000_iter/%04d.png' % k)

    if k == 1437:
      break





