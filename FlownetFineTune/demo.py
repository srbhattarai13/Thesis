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

def np_load_frame(frame1, frame2, resize_height=256, resize_width=256):
    im1 = imread(frame1)
    im2 = imread(frame2)

    im1_resized = imresize(im1, (resize_width, resize_height))
    im2_resized = imresize(im2, (resize_width, resize_height))

    ims = np.array([[im1_resized, im2_resized]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)

    ims_tensor = torch.from_numpy(ims)

    ims_normalize = (ims_tensor / 127.5) - 1.0

    ims_v = Variable(ims_tensor.cuda(), requires_grad=True)

    return ims_v



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--rgb_max", type=float, default=255.)
  args = parser.parse_args()


  im1 = imread('imgs/0381.jpg')
  im1_resized = imresize(im1, (256,256))
  im2 = imread('imgs/0382.jpg')
  im2_resized = imresize(im2, (256,256))


  # B x 3(RGB) x 2(pair) x H x W
  ims = np.array([[im1_resized, im2_resized]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)

  ims = torch.from_numpy(ims)
  ims_v = Variable(ims.cuda(), requires_grad=False)


  # Build model
  flownet2 = FlowNet2SD(args)
  model_path = '../flownet2-pytorch/pretrained/FlowNet2-SD_checkpoint.pth.tar'
  pretrained_dict = torch.load(model_path)['state_dict']
  # model_dict = flownet2.state_dict()
  # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  # model_dict.update(pretrained_dict)
  flownet2.load_state_dict(pretrained_dict)




  print(flownet2)
  # flownet2.cuda()
  # flownet2.eval()
  # pred_flow = flownet2(ims_v).cpu().data
  # pred_flow = pred_flow[0].numpy().transpose((1, 2, 0))
  # flow_im = flow_to_image(pred_flow)

  # Visualization
  # plt.imshow(flow_im)
  # plt.savefig('imgs/trained_check.png', bbox_inches='tight')


  # model = torch.load(path)
  # model.eval()
  # # path = 'checkpoints/batch_64_epoch_0_Iteration_3_1e-05_fnet_nextPict_ShTech.pt'
  # pretrained_dict = torch.load(path)['state_dict']
  #
  #
  # model = torch.load(path)
  # model_dict = flownet2.state_dict()
  # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  # model_dict.update(pretrained_dict)
  # flownet2.load_state_dict(model_dict)
  # flownet2.cuda()
  # # model.cuda()
  # # pred_flow = model(ims_v).cpu().data
  # # pred_flow = model(ims_v).cpu().data
  # prediction = flownet2(ims_v).cpu().data
  # # pred_flow = model(ims_v).cpu().data
  # #
  # # print(prediction.size())
  #
  # # print(len(pred_flow))
  #
  # # # #
  # # pred_flow = pred_flow[0].numpy().transpose((1,2,0))
  # prediction = prediction[0].numpy().transpose((1,2,0))
  # flow_im = flow_to_image(prediction)
  #
  # # flow_im = flow_to_image(pred_flow)
  # image = Image.fromarray(flow_im, 'RGB')
  # image = image.resize(( int(scale_width * adapted_width),int(scale_height * adapted_height)), Image.ANTIALIAS)
  # image.save('imgs/RandomWeight.png')
  # # image.show()
  # # print(flow_im.shape)
  # # flow_im = flow_im.reshape((int(scale_height * adapted_height), int(scale_width * adapted_width), flow_im.shape[2]))
  # # print(flow_im.shape)


  # # Visualization
  # plt.figure(figsize=(360, 840))
  # plt.plot(flow_im)
  # plt.show()
  # # plt.imshow(flow_im)
  # plt.savefig('imgs/saving_orgsize_12.png', bbox_inches='tight')