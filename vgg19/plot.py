import torch
import matplotlib.pyplot as plt
import numpy as np
import pylab


feats = '/media/data/Shreeram/datasets/shanghaitech/training/vggfeatures/01_002/0710.npy'


nparray = np.load(feats).squeeze()

# print(nparray.shape[0])
# fristFeat = nparray[511,:,:]
# # print(fristFeat.shape)
# plt.imshow(fristFeat)
# plt.show()
#
# # num_plot = 100
# # fig = plt.figure(figsize=(100,100))
# fig, axarr = plt.subplots(min(nparray.shape[0],num_plot))
# for idx in range(min(nparray.shape[0],num_plot)):
#     fig.add_subplot(10,10,idx+1)
#     plt.imshow(nparray[idx,:,:])
#
# plt.show()


square = 20
ix = 1
for _ in range(square):
	for _ in range(square):

		# specify subplot and turn of axis
		ax = plt.subplot(square, square,ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(nparray[ix, :, :], cmap='gray')
		ix += 1

# show the figure
plt.show()