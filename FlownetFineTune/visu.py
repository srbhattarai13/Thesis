#
# from torchvision import utils
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
#   '''
#   vistensor: visuzlization tensor
#       @ch: visualization channel
#       @allkernels: visualization all tensores
#   '''
#
#   n ,c ,w ,h = tensor.shape
#   if allkernels: tensor = tensor.view( n *c ,-1 ,w ,h )
#   elif c != 3: tensor = tensor[: ,ch ,: ,:].unsqueeze(dim=1)
#
#   rows = np.min( (tensor.shape[0 ]/ /nrow + 1, 64 )  )
#   grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
#   plt.figure( figsize=(nrow ,rows) )
#   plt.imshow(grid.numpy().transpose((1, 2, 0)))
#
# def savetensor(tensor, filename, ch=0, allkernels=False, nrow=8, padding=2):
#   '''
#   savetensor: save tensor
#       @filename: file name
#       @ch: visualization channel
#       @allkernels: visualization all tensores
#   '''
#
#   n ,c ,w ,h = tensor.shape
#   if allkernels: tensor = tensor.view( n *c ,-1 ,w ,h )
#   elif c != 3: tensor = tensor[: ,ch ,: ,:].unsqueeze(dim=1)
#   utils.save_image(tensor, filename, nrow=nrow )
#
#
# ik = 0
# # kernel = alexnet.features[ik].weight.data.clone()
# # print(kernel.shape)
#
# vistensor(kernel, ch=0, allkernels=False)
# savetensor(kernel ,'kernel.png', allkernels=False)
#
# plt.axis('off')
# plt.ioff()
# plt.show()
#
#
import re
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.misc import imread
import torch
from torch.autograd import Variable
from scipy.misc import imresize
from math import ceil
from math import floor

divisor = 64
# from models import FlowNet2SD
# from FlowNet2_src import flow_to_image
import matplotlib.pyplot as plt
from flowsd import FlowNet2SD

from PIL import Image
import argparse, os, sys, subprocess
UNKNOWN_FLOW_THRESH = 1e7


def readPFM(file):
  file = open(file, 'rb')

  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().rstrip()
  if header.decode("ascii") == 'PF':
    color = True
  elif header.decode("ascii") == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
  if dim_match:
    width, height = list(map(int, dim_match.groups()))
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().decode("ascii").rstrip())
  if scale < 0:  # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>'  # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)

  data = np.reshape(data, shape)
  data = np.flipud(data)
  return data, scale


def readFlow(name):
  if name.endswith('.pfm') or name.endswith('.PFM'):
    return readPFM(name)[0][:, :, 0:2]

  f = open(name, 'rb')

  header = f.read(4)
  if header.decode("utf-8") != 'PIEH':
    raise Exception('Flow file header does not contain PIEH')

  width = np.fromfile(f, np.int32, 1).squeeze()
  height = np.fromfile(f, np.int32, 1).squeeze()

  flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

  return flow.astype(np.float32)


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


flow = readFlow('/media/data/Shreeram/datasets/shanghaitech/training/flow/01_001/0000.flo')#[0]
print(flow.shape)
# flow_im = flow_to_image(flow)
# plt.imshow(flow_im)
# plt.savefig('Flow.png',bbox_inches='tight')
