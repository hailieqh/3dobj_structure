from __future__ import division
import os
import math
import random
import copy
import scipy.io
import h5py
from skimage import io, transform
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset
from opts import *
from utils import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class MapDataset(object):
	def __init__(self, root_dir, transform=None, datacat='chair', phase='train'):
		global opt
		opt = get_opt()
		if phase == 'train':
			self.num = [5000, 10000, 15000, 20000]
		else:
			self.num = [250, 500, 750, 1000]
		self.root_dir = os.path.join(root_dir, datacat, 'map') # root_dir: /data/chair/map/
		self.gt = load_noise_gt(self.root_dir, datacat, phase)
		self.input = self.load_input(self.root_dir, datacat, phase)
		self.transform = transform
		self.datacat = datacat
		self.phase = phase

	def __len__(self):
		return self.gt.shape[0]

	def __getitem__(self, idx):
		label = self.gt[idx]
		if idx >= 0 and idx < self.num[0]:
			heatmap = copy.deepcopy(self.input[self.num[0]][idx])
		elif idx >= self.num[0] and idx < self.num[1]:
			heatmap = copy.deepcopy(self.input[self.num[1]][idx - self.num[0]])
		elif idx >= self.num[1] and idx < self.num[2]:
			heatmap = copy.deepcopy(self.input[self.num[2]][idx - self.num[1]])
		elif idx >= self.num[2] and idx < self.num[3]:
			heatmap = copy.deepcopy(self.input[self.num[3]][idx - self.num[2]])
		label = torch.from_numpy(label)
		heatmap = torch.from_numpy(heatmap)
		sample = {'image': heatmap, 'label': label}
		if self.transform:
			sample = self.transform(sample)
		return sample

	def load_input(self, root_dir, datacat, phase):
		inputmap = {}
		for n in self.num:
			inputmap_file = phase + 'map_{}'.format(n) + '.mat'
			inputmap_dir = os.path.join(root_dir, inputmap_file)
			inputmap_tmp = scipy.io.loadmat(inputmap_dir)
			inputmap[n] = inputmap_tmp['outputs']	# (30000, 14)
		return inputmap




class SYNDataset(Dataset):
	def __init__(self, root_dir, transform=None, datacat='chair', phase='train'):
		global opt
		opt = get_opt()
		self.root_dir = os.path.join(root_dir, datacat) # root_dir: /data/chair
		self.gt = self.load_gt(self.root_dir, datacat, phase)
		self.transform = transform
		self.datacat = datacat
		self.phase = phase

	def __len__(self):
		return self.gt.shape[0]

	def __getitem__(self, idx):
		label = self.gt[idx]
		label_unnorm = unnorm(self.gt[idx])
		heatmap, _ = getX2D(label_unnorm, self.datacat, opt.interRes, opt.interRes, 
						opt.outputRes, opt.outputRes, True, opt.hmGauss)#, noise=True)
		label = torch.from_numpy(label)
		heatmap = torch.from_numpy(heatmap.transpose((2, 0, 1)))
		sample = {'image': heatmap, 'label': label}
		if self.transform:
			sample = self.transform(sample)
		return sample

	def load_gt(self, root_dir, datacat, phase):
		gt_file = datacat + '_gt_' + phase + '.mat'
		gt_dir = os.path.join(root_dir, gt_file)
		gt_tmp = scipy.io.loadmat(gt_dir)
		gt = gt_tmp['outputs']	# (30000, 14)
		return gt

def load_noise_gt(root_dir, datacat, phase):
	gt_file = datacat + '_gtn_' + phase + '.mat'
	gt_dir = os.path.join(root_dir, gt_file)
	gt_tmp = scipy.io.loadmat(gt_dir)
	gt = gt_tmp['outputs']	# (20000, 14)
	return gt

def save_heatmap(root_dir, datacat, phase):
	global opt
	opt = get_opt()
	print(opt.nmiss)
	if not os.path.exists('map'):
		os.mkdir('map')
	root_dir = os.path.join(root_dir, datacat, 'map')
	gt = load_noise_gt(root_dir, datacat, phase)
	num, _ = gt.shape
	if phase == 'train':
		inputmap = np.zeros((1, opt.nKeypoints, opt.outputRes, opt.outputRes))
	else:
		inputmap = np.zeros((num, opt.nKeypoints, opt.outputRes, opt.outputRes))



	# div_2 = num // 2
	# div_4 = num // 4
	# div_8 = num // 8
	div_40 = num // 40
	# if opt.dataset == 'car':
	# 	div = div_8
	# else:
	# 	div = div_4
	# inputmap = np.zeros((div, opt.nKeypoints, opt.outputRes, opt.outputRes))
	# file_j = 0
	for i in xrange(num):
		label = gt[i]
		label_unnorm = unnorm(gt[i])
		nodenum = 0
		noisy = False
		# if i < div_2:
		# 	pass # save clean maps
		# 	#noisy = True
		# 	#nodenum = i // div_8 + 1 # 1, 2, 3, 4
		# if i >= div_2:
		# 	m = i - div_2
		# 	noisy = True
		# 	nodenum = m // div_8 + 1 # 1, 2, 3, 4
		heatmap, _ = getX2D(label_unnorm, datacat, opt.interRes, opt.interRes,
						opt.outputRes, opt.outputRes, True, opt.hmGauss, noise=noisy, node=nodenum)
		if phase == 'train':
			inputmap[0] = heatmap.transpose((2, 0, 1))
			scipy.io.savemat('map/' + phase + 'map_{}.mat'.format(i), {'outputs': inputmap})
			inputmap = np.zeros((1, opt.nKeypoints, opt.outputRes, opt.outputRes))
		else:
			inputmap[i] = heatmap.transpose((2, 0, 1))



		# inputmap[i - file_j*div] = heatmap.transpose((2, 0, 1))
		# if (i+1) % div == 0:
		# 	scipy.io.savemat('map/' + phase + 'map_{}.mat'.format(i+1), {'outputs': inputmap})
		# 	inputmap = np.zeros((div, opt.nKeypoints, opt.outputRes, opt.outputRes))
		# 	file_j += 1

		if i % div_40 == 0:
			outmap = copy.deepcopy(heatmap[:, :, 0])
			for p in xrange(1, opt.nKeypoints):
				outmap += copy.deepcopy(heatmap[:, :, p])
			fig = plt.figure()
			plt.imshow(outmap)
			plt.savefig('out/{}.png'.format(i))
			plt.close('all')
			print(i)

	if phase == 'val':
		scipy.io.savemat('map/' + phase + 'map_{}.mat'.format(i+1), {'outputs': inputmap})
	print(opt.nmiss)