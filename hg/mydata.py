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

class KPDataset(Dataset):
	def __init__(self, gt_file, root_dir, transform=None, datacat='bed', phase='train'):
		global opt
		opt = get_opt()
		self.root_dir = os.path.join(root_dir, datacat) # root_dir: /data/bed
		self.gt = self.load_gt(self.root_dir, gt_file)	# (2, 30, 4, 1480)
		self.transform = transform
		self.datacat = datacat
		self.phase = phase
		self.image_idx = []
		self.image_idx = self.load_id(self.root_dir, self.phase)
		self.sigma = opt.hmGauss
		self.gauss_kernel = gaussian2D(6*self.sigma+1, self.sigma)

	def __len__(self):
		return len(self.image_idx)

	def __getitem__(self, idx):
		# idx: 0, ..., self.num - 1
		or_idx = self.image_idx[idx]
		image = self.load_image(or_idx)
		num_kp = opt.nKeypoints
		#label = np.median(self.gt[:, :num_kp, :, or_idx - 1], axis=2)
		nperson = self.gt.shape[2] # 3 or 4
		if len(image.shape) == 2:
			image = np.array([image, image, image]).transpose((1, 2, 0)) # gray image
		h, w, c = image.shape
		label_mean = self.get_mean(num_kp, nperson, or_idx)
		label_std = self.get_std(num_kp, nperson, or_idx, label_mean, h, w)
		label = self.get_median(num_kp, nperson, or_idx)
		label = label.transpose()
		label_std = label_std.transpose()
		#visualize_img_1(image, label, or_idx)
		sample = {'image': image, 'label': label} # label (10, 2)
		if self.transform:
			sample = self.transform(sample)
		#visualize_img_2(sample['image'], sample['label'], or_idx)
		sample['label'] = draw_gaussian(sample['label'], self.sigma, self.gauss_kernel)
		sample['label'] = sample['label'].transpose((2, 0, 1))
		sample['label'] = torch.from_numpy(sample['label'])

		return (sample, label_std)

	def load_flic(self, root_dir, phase):
		f = h5py.File(root_dir)


	def load_gt(self, root_dir, gt_file):
		gt_dir = os.path.join(root_dir, gt_file)
		gt_tmp = scipy.io.loadmat(gt_dir)
		gt = gt_tmp['coords']	# (2, 30, 4, 1480)
		return gt

	def load_id(self, root_dir, phase):
		if opt.dataset == 'car' and opt.test:
			phase_file_dir = root_dir + '/test.txt'
		else:
			phase_file_dir = root_dir + '/' + phase + '.txt'
		f = open(phase_file_dir, 'r')
		lines = f.readlines()
		image_idx = []
		for line in lines:
			idx = int(line[-9:-5])
			image_idx.append(idx)
		return image_idx

	def load_image(self, or_idx):
		tmp_name = '{}'.format(or_idx)
		image_name = tmp_name
		for i in xrange(8 - len(tmp_name)):
			image_name = '0' + image_name
		if opt.dataset == 'car':
			image_name = image_name + '.png'
		else:
			image_name = image_name + '.jpg'
		image_name = os.path.join(self.root_dir, 'images', image_name)
		image = io.imread(image_name)
		return image

	def get_std(self, num_kp, nperson, or_idx, label_mean, h, w):
		tmp = np.zeros(self.gt[0, :num_kp, 0, or_idx - 1].shape)
		count = 0
		for i in xrange(nperson):
			if np.sum(np.abs(self.gt[:, :num_kp, i, or_idx - 1])) > 0:
				if w > h:
					p = w
				else:
					p = h######padding
				x = (self.gt[0, :num_kp, i, or_idx - 1] - label_mean[0]) * 64.0 / p#w  ###cut??
				y = (self.gt[1, :num_kp, i, or_idx - 1] - label_mean[1]) * 64.0 / p#h
				tmp += x**2 + y**2
				count += 1
		label_std = np.sqrt(tmp / count) + 0.00001
		return label_std

	def get_mean(self, num_kp, nperson, or_idx):
		label_mean = np.zeros((2, num_kp))
		count = 0
		for i in xrange(nperson):
			if np.sum(np.abs(self.gt[:, :num_kp, i, or_idx - 1])) > 0:
				label_mean += self.gt[:, :num_kp, i, or_idx - 1]
				count += 1
		label_mean /= count
		if count == 0:
			print label_mean, count
			return
		return label_mean

	def get_median(self, num_kp, nperson, or_idx):
		label = np.zeros((2, num_kp))
		for i in xrange(2):
			for j in xrange(num_kp):
				tmp = []
				count = 0
				for k in xrange(nperson):
					if np.abs(self.gt[i, j, k, or_idx - 1]) > 0:
						tmp.append(self.gt[i, j, k, or_idx - 1])
						count += 1
				if count > 0:
					tmp.sort()
					label[i, j] = tmp[int(count/2)]
		return label

class FLICDataset(Dataset):
	def __init__(self, gt_file, root_dir, transform=None, datacat='flic', phase='train'):
		global opt
		opt = get_opt()
		self.gt_file = phase + '.h5'
		self.root_dir = os.path.join(root_dir, datacat) # root_dir: /data/flic
		self.transform = transform
		self.datacat = datacat
		self.phase = phase
		self.center = None # 3987, 2
		self.imgname = None # 3987
		self.index = None # 3987
		self.normalize = None # 3987
		self.part = None # 3987, 11, 2
		self.person = None # 3987
		self.scale = None # 3987
		self.torsoangle = None # 3987
		self.visible = None # 3987, 11
		self.sigma = opt.hmGauss
		self.gauss_kernel = gaussian2D(6*self.sigma+1, self.sigma)
		self.load_h5(self.root_dir, self.gt_file)

	def __len__(self):
		return len(self.index)

	def __getitem__(self, idx):
		# idx: 0, ..., self.num - 1
		name = self.imgname[idx]
		img_dir = os.path.join(self.root_dir, 'images', name)
		image = io.imread(img_dir)
		num_kp = opt.nKeypoints
		if len(image.shape) == 2:
			image = np.array([image, image, image]).transpose((1, 2, 0)) # gray image
		h, w, c = image.shape
		label = copy.deepcopy(self.part[idx])
		#visualize_img_1(image, label, idx)
		sample = {'image': image, 'label': label} # label (11, 2)
		center = copy.deepcopy(self.center[idx])
		scale = copy.deepcopy(self.scale[idx])
		sample, torso = self.crop(sample, center, scale, idx)
		if self.transform:
			sample = self.transform(sample)
		#visualize_img_2(sample['image'], sample['label'], idx)
		sample['label'] = draw_gaussian(sample['label'], self.sigma, self.gauss_kernel)
		sample['label'] = sample['label'].transpose((2, 0, 1))
		sample['label'] = torch.from_numpy(sample['label'])


		return (sample, np.ones(11)*torso)

	def load_h5(self, root_dir, gt_file):
		gt_dir = os.path.join(root_dir, 'annot', gt_file)
		f = h5py.File(gt_dir, 'r')
		self.center = f[u'center'][()] # 3987, 2
		self.imgname = f[u'imgname'][()] # 3987
		self.index = f[u'index'][()] # 3987
		#self.normalize = f[u'normalize'][()] # 3987
		self.part = f[u'part'][()] # 3987, 11, 2
		self.person = f[u'person'][()] # 3987
		self.scale = f[u'scale'][()] # 3987
		#self.torsoangle = f[u'torsoangle'][()] # 3987
		#self.visible = f[u'visible'][()] # 3987, 11

	def crop(self, sample, center, scale, idx):
		image, label = sample['image'], sample['label']
		h, w, c = image.shape
		if self.phase == 'train':
			scale *= (1 + random.random() * 0.1)#0.95
		new_h, new_w = int(h / scale), int(w / scale)
		image = transform.resize(image, (new_h, new_w))
		label[:, 0] *= new_w / w
		label[:, 1] *= new_h / h
		center[0] *= new_w / w
		center[1] *= new_h / h
		center[1] += 5
		size = 300
		left = int(center[0] - size /2)
		right = int(center[0] + size /2)
		top = int(center[1] - size / 2)
		down = int(center[1] + size / 2)
		l_shift = 0
		t_shift = 0
		if left < 0:
			l_shift = -left
			left = 0
		if right > new_w:
			right = new_w
		if top < 0:
			t_shift = -top
			top = 0
		if down > new_h:
			down = new_h
		imageout = np.zeros((size, size, c))
		imageout[t_shift : down-top+t_shift, l_shift : right-left+l_shift, :] = copy.deepcopy(
			image[top : down, left : right, :])*255
		imageout = imageout.astype(np.uint8)
		label = label - [left, top] + [l_shift, t_shift]
		image = imageout
		if self.phase == 'train':
			image, label = rotate(imageout, label, size)
		outres = 200
		imageout = np.zeros((outres, outres, c))
		start = int((size - outres) / 2.0)
		end = int(start + outres)
		imageout[:, :, :] = image[start : end, start : end, :]
		label = label - [start, start]
		imageout = imageout.astype(np.uint8)

		torso = np.linalg.norm((label[3, :] - label[6, :]) * 64.0 / 200.0)
		return {'image': imageout, 'label': label}, torso

def rotate(img, pts, size):
	a = random.random()
	b = random.random()
	if a < 0.5:
		rot = 15 * b
	else:
		rot = -15 * b
	center = np.ones(2) * size / 2
	num = pts.shape[0]
	outpts = np.zeros(pts.shape)
	r = np.eye(3)
	ang = -rot * math.pi / 180
	s = math.sin(ang)
	c = math.cos(ang)
	r[0][0] = c
	r[0][1] = -s
	r[1][0] = s
	r[1][1] = c
	for i in xrange(num):
		pt = np.ones(3)
		pt[0], pt[1] = pts[i][0]-center[0], pts[i][1]-center[1]
		new_point = np.dot(r, pt)
		outpts[i] = new_point[0:2] + center
	img = Image.fromarray(np.uint8(img))
	img = img.rotate(rot)
	img = np.asarray(img)

	return img, outpts


def gaussian2D(size, sigma):
		sigma_2 = 2 * sigma * sigma
		r_size = (size - 1)//2
		gk = np.zeros((size, size))
		for i in range(-r_size, r_size + 1):
			h = i + r_size
			for j in range(-r_size, r_size + 1):
				w = j + r_size
				v = np.exp(-(i*i + j*j) / sigma_2)
				gk[h, w] = v
				if i*i + j*j > r_size*r_size:
					gk[h, w] = 0
		return gk

def draw_gaussian(label, sigma, gauss_kernel):
		num_kp = opt.nKeypoints # 10
		h, w = opt.outputRes, opt.outputRes # 64, 64
		out = np.zeros((h, w, num_kp)) # (64, 64, 10)
		for i in xrange(num_kp):
			x, y = label[i]
			if x < 0 or x >= 64 or y < 0 or y >= 64:
				continue
			x, y = int(x), int(y)
			r_size = 3 * sigma
			tmp = np.zeros((64 + 2 * r_size, 64 + 2 * r_size))
			tmp[y : y+2*r_size+1, x : x+2*r_size+1] = gauss_kernel
			out[:, :, i] = tmp[r_size : 64+r_size, r_size : 64+r_size]
		return out

def draw_gaussian0(label):
		num_kp = opt.nKeypoints # 10
		h, w = opt.outputRes, opt.outputRes # 64, 64
		out = np.zeros((h, w, num_kp)) # (64, 64, 10)
		sigma = opt.hmGauss / 10.0 # 1 / 10
		#tmpSize = math.ceil(3*sigma) # 3
		X = np.linspace(-w, w, 2 * w + 1)
		Y = np.linspace(-h, h, 2 * h + 1)
		xx, yy = np.meshgrid(X, Y)
		dist = xx * xx + yy * yy
		gauss = np.exp(-sigma * dist).astype(np.float32)
		for i in xrange(num_kp):
			x, y = label[i]
			if x < 0 or x > 64 or y < 0 or y > 64:
				continue
			delY, delX = h - y, w - x
			xSt, ySt = max(0, delX), max(0, delY)
			yEn, xEn = delY + h, delX + w
			yEn , xEn = min(2 * h + 1, yEn), min(2 * w + 1, xEn)
			yImSt, xImSt = max(0, y - h), max(0, x - w)
			yImEn, xImEn = yImSt + (yEn - ySt), xImSt + (xEn - xSt)
			if yImSt < 0 or yImEn > h or xImSt < 0 or xImEn > w:
				continue
			if ySt < 0 or yEn > 2*h or xSt < 0 or xEn > 2*w:
				continue
			yImSt, yImEn, xImSt, xImEn = int(yImSt), int(yImEn), int(xImSt), int(xImEn)
			ySt, yEn, xSt, xEn = int(ySt), int(yEn), int(xSt), int(xEn)
			if (yEn-ySt) != 64:
				delta = 64-(yEn-ySt)
				if ySt-delta > 0:
					ySt -= delta
				else:
					yEn += delta
			if (xEn-xSt) != 64:
				delta = 64-(xEn-xSt)
				if xSt-delta > 0:
					xSt -= delta
				else:
					xEn += delta
			if (yImEn-yImSt) != 64:
				delta = 64-(yImEn-yImSt)
				if yImSt-delta > 0:
					yImSt -= delta
				else:
					yImEn += delta
			if (xImEn-xImSt) != 64:
				delta = 64-(xImEn-xImSt)
				if xImSt-delta > 0:
					xImSt -= delta
				else:
					xImEn += delta
			out[yImSt:yImEn, xImSt:xImEn, i] = copy.deepcopy(gauss[ySt:yEn, xSt:xEn])
			#plt.imshow(out[:,:,i], cmap=cm.gray)
			#plt.show()

			# ul = (math.floor(label[i][0] - tmpSize), math.floor(label[i][1] - tmpSize)) # lefttop
			# br = (math.floor(label[i][0] + tmpSize), math.floor(label[i][1] + tmpSize)) # rightdown
			# if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
			#     continue
			# size = 2 * tmpSize + 1 # 7
			# g_x = (math.max(0, -ul[0]), 
			#         math.min(br[0], w) - math.max(0, ul[0]) + math.max(0, -ul[0]))
			# g_y = (math.max(0, -ul[1]), 
			#         math.min(br[1], h) - math.max(0, ul[1]) + math.max(0, -ul[1]))
			# out_x = (math.max(0, ul[0]), math.min(br[0], w - 1))
			# out_y = (math.max(0, ul[1]), math.min(br[1], h - 1))
			# assert g_x[0] > 0 and g_y[1] > 0
			# out[:, :, i] = copy.deepcopy()
		return out

class PadSquare(object):
	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		h, w, c = image.shape
		tmp = image
		if h < w:
			pad = (w - h) // 2
			tmp = np.zeros((w, w, c))
			tmp[pad : pad+h, :, :] = copy.deepcopy(image)
			label[:, 1] += pad
		elif h > w:
			pad = (h - w) // 2
			tmp = np.zeros((h, h, c))
			tmp[:, pad : pad+w, :] = copy.deepcopy(image)
			label[:, 0] += pad
		image = tmp.astype(np.uint8)
		return {'image': image, 'label': label}

class RandomRotate(object):
	def __call__(self, sample):
		img, pts = sample['image'], sample['label']
		size = img.shape[0]
		a = random.random()
		b = random.random()
		if a < 0.5:
			rot = 15 * b
		else:
			rot = -15 * b
		center = np.ones(2) * size / 2
		num = pts.shape[0]
		outpts = np.zeros(pts.shape)
		r = np.eye(3)
		ang = -rot * math.pi / 180
		s = math.sin(ang)
		c = math.cos(ang)
		r[0][0] = c
		r[0][1] = -s
		r[1][0] = s
		r[1][1] = c
		for i in xrange(num):
			pt = np.ones(3)
			pt[0], pt[1] = pts[i][0]-center[0], pts[i][1]-center[1]
			new_point = np.dot(r, pt)
			outpts[i] = new_point[0:2] + center
		img = Image.fromarray(np.uint8(img))
		img = img.rotate(rot)
		img = np.asarray(img)

		return {'image': img, 'label': outpts}

class RandomHorizontalFlip(object):
	"""Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

	def __call__(self, sample):
		"""
		Args:
			img (PIL.Image): Image to be flipped.
		Returns:
			PIL.Image: Randomly flipped image.
		"""
		img, label = sample['image'], sample['label']
		h, w, c = img.shape
		#if True:
		if random.random() < 0.5:
			img = Image.fromarray(np.uint8(img))
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
			sample['image'] = np.asarray(img)
			label[:, 0] = w - label[:, 0]
			n = opt.nKeypoints
			nn = int(n / 2)
			if opt.dataset == 'bed' or opt.dataset == 'sofa' or opt.dataset == 'car':
				tmp = copy.deepcopy(label[0:nn, :])
				label[0:nn, :] = copy.deepcopy(label[nn:n, :])
				label[nn:n, :] = copy.deepcopy(tmp)
			if opt.dataset == 'chair' or opt.dataset == 'table':
				for i in xrange(nn):
					j = i * 2
					tmp = copy.deepcopy(label[j, :])
					label[j, :] = copy.deepcopy(label[j+1, :])
					label[j+1, :] = copy.deepcopy(tmp)
			if opt.dataset == 'swivelchair':
				inter = [4, 2]
				for i in xrange(2):
					tmp = copy.deepcopy(label[i, :])
					label[i, :] = copy.deepcopy(label[i+inter[i], :])
					label[i+inter[i], :] = copy.deepcopy(tmp)
				for i in xrange(3):
					j = i * 2 + 7
					tmp = copy.deepcopy(label[j, :])
					label[j, :] = copy.deepcopy(label[j+1, :])
					label[j+1, :] = copy.deepcopy(tmp)
			if opt.dataset == 'flic':
				for i in xrange(3):
					tmp = copy.deepcopy(label[i, :])
					label[i, :] = copy.deepcopy(label[i+3, :])
					label[i+3, :] = copy.deepcopy(tmp)
				for i in xrange(2):
					j = i * 2 + 6
					tmp = copy.deepcopy(label[j, :])
					label[j, :] = copy.deepcopy(label[j+1, :])
					label[j+1, :] = copy.deepcopy(tmp)
			sample['label'] = label
		return sample

class RandomCut(object):
	def __init__(self, edge):
		self.edge = edge

	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		h, w, c = image.shape
		n, d = label.shape
		edge = int(self.edge * random.random())
		if w >= h:
			left = label[:, 0] - 2*edge
			right = w - (label[:, 0] + 2*edge)
			if random.random() < 0.5 and np.sum(np.abs(left)) == np.sum(left):
				image = image[:, edge:, :]
				label[:, 0] = label[:, 0] - edge
			if random.random() < 0.5 and np.sum(np.abs(right)) == np.sum(right):
				image = image[:, :w-edge, :]
				label[:, 0] = label[:, 0] + edge
		else:
			up = label[:, 1] - 2*edge
			down = h - (label[:, 1] + 2*edge)
			if random.random() < 0.5 and np.sum(np.abs(up)) == np.sum(up):
				image = image[edge:, :, :]
				label[:, 1] = label[:, 1] - edge
			if random.random() < 0.5 and np.sum(np.abs(down)) == np.sum(down):
				image = image[:h-edge, :, :]
				label[:, 1] = label[:, 1] + edge
		sample['image'], sample['label'] = image, label

		return sample

class RandomRColor(object):
	def __call__(self, sample):
		img, _ = sample['image'], sample['label']
		tmp = np.zeros(img.shape)
		tmp[:, :, 0] = copy.deepcopy(img[:, :, 0])*np.random.uniform(0.6,1.4)
		tmp[:, :, 1] = copy.deepcopy(img[:, :, 1])*np.random.uniform(0.6,1.4)
		tmp[:, :, 2] = copy.deepcopy(img[:, :, 2])*np.random.uniform(0.6,1.4)
		tmp = np.maximum(tmp, 0)
		tmp = np.minimum(tmp, 255)
		tmp = tmp.astype(np.uint8)
		sample['image'] = tmp
		return sample


class Rescale(object):
	"""Rescale the image in a sample to a given size.

	Args:
		output_size (tuple or tuple): Desired output size. If tuple, output is
			matched to output_size. If int, smaller of image edges is matched
			to output_size keeping aspect ratio the same.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['label']

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		img = transform.resize(image, (new_h, new_w))

		# h and w are swapped for landmarks because for images,
		# x and y axes are axis 1 and 0 respectively
		#landmarks = landmarks * [new_w / w, new_h / h]
		landmarks[:, 0] *= 64.0/w
		landmarks[:, 1] *= 64.0/h

		return {'image': img, 'label': landmarks}

class RandomCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['label']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h,
					  left: left + new_w,
					  :]

		landmarks = landmarks - [left, top]

		return {'image': image, 'label': landmarks}


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['label']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image),
				'label': torch.from_numpy(landmarks)}