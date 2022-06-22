from __future__ import print_function, division
import os
import math
import copy
import random
import torch
import scipy.io
import pandas as pd
from skimage import io, transform
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from torchvision import transforms, utils
from opts import *


def unnorm(output):
	global opt
	opt = get_opt()
	if opt.dataset == 'chair':
		mul_mask = np.array([20, 170, 170, 65, 15, 1/350-1/450, 2, 2, 2, 2, 2, 2, 120, 160])
		add_mask = np.array([30, -70, -70, -25, 0, 1/450, -1, -1, -1, -1, -1, -1, -60, -80])
	elif opt.dataset == 'sofa':
		mul_mask = np.array([12, 42, 84, 63, 126, 84, 64, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		add_mask = np.array([30, -11, 0, -21, -62, -42, 0, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
	elif opt.dataset == 'swivelchair':
		mul_mask = np.array([50, 105, 257.5, 107.5, 107.5, 76.5, 51, 1/350-1/450, 2, 2, 2, 2, 2, 2, 120, 160])
		add_mask = np.array([25, -42.5, -95, -42.5, -42.5, -25.5, 0, 1/450, -1, -1, -1, -1, -1, -1, -60, -80])
	elif opt.dataset == 'table':
		# mul_mask = np.array([40, 100, 100, 10, 50, 50, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		# add_mask = np.array([30, -50, 0, 0, -25, -25, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
		# mul_mask = np.array([40, 50, 124, 70, 40, 40, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		# add_mask = np.array([30, -25, -24, -10, -20, -20, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
		# mul_mask = np.array([30, 50, 124, 42, 20, 20, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		# add_mask = np.array([40, -25, -24, -12, -10, -10, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
		mul_mask = np.array([30, 50, 124, 100, 40, 40, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		add_mask = np.array([30, -25, -24, -10, -20, -20, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
	elif opt.dataset == 'car':
		# mul_mask = np.array([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		# add_mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/650, -1, -1, -1, -1, -1, -1, -80, -60])
		mul_mask = np.array([1, 1, 1, 1, 1, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		add_mask = np.array([0, 0, 0, 0, 0, 1/650, -1, -1, -1, -1, -1, -1, -80, -60])#pca
	output = output * mul_mask + add_mask
	return output

def perturb(x, node):
	num_kp = opt.nKeypoints
	idxs = range(opt.nKeypoints)
	random.shuffle(idxs)
	mean = [0, 0]
	cov = [[1, 0], [0, 1]]
	shift = np.random.multivariate_normal(mean, cov, opt.nKeypoints) * 3#10
	# print('-------------------------------')
	# print('x', x)
	# print('shift', shift)
	for i in xrange(opt.nKeypoints):
		if i < node:
			x[:, idxs[i]] = -x[:, idxs[i]]
		else:
			x[:, idxs[i]] = x[:, idxs[i]] + shift[i, :].T.astype(int)
	# print('x', x)
	# print('-------------------------------')
	return x



def perturb0(x, node):
	num_kp = opt.nKeypoints
	idxs = np.array([-1, -1, -1, -1])
	curidx = 0
	while True:
		if curidx > 3:
			break
		tmp = np.random.randint(num_kp, size=1)
		flag = 0
		for i in xrange(curidx):
			if tmp[0] == idxs[i]:
				flag += 1
		if flag == 0:
			idxs[curidx] = tmp[0]
			curidx += 1
	# temp = set(idxs)
	# if len(temp) == len(list(idxs)):
	# 	opt.nmiss[7] += 1
	mean = [0, 0]
	cov = [[1, 0], [0, 1]]
	shift = np.random.multivariate_normal(mean, cov, 4) * 4
	# if node >= 3:
	# 	print('-------------------------------')
	# 	print('x', x)
	# 	print('shift', shift)
	if node == 0:
		#opt.nmiss[node] += 1
		x[:, idxs[0]] = x[:, idxs[0]] + shift[0, :].T.astype(int)
		x[:, idxs[1]] = x[:, idxs[1]] + shift[1, :].T.astype(int)
		x[:, idxs[2]] = x[:, idxs[2]] + shift[2, :].T.astype(int)
		x[:, idxs[3]] = x[:, idxs[3]] + shift[3, :].T.astype(int)
	elif node == 1:
		#opt.nmiss[node] += 1
		x[:, idxs[0]] = -x[:, idxs[0]]
		x[:, idxs[1]] = x[:, idxs[1]] + shift[1, :].T.astype(int)
		x[:, idxs[2]] = x[:, idxs[2]] + shift[2, :].T.astype(int)
		x[:, idxs[3]] = x[:, idxs[3]] + shift[3, :].T.astype(int)
	elif node == 2:
		#opt.nmiss[node] += 1
		x[:, idxs[0]] = -x[:, idxs[0]]
		x[:, idxs[1]] = -x[:, idxs[1]]
		x[:, idxs[2]] = x[:, idxs[2]] + shift[2, :].T.astype(int)
		x[:, idxs[3]] = x[:, idxs[3]] + shift[3, :].T.astype(int)
	elif node == 3:
		#opt.nmiss[node] += 1
		x[:, idxs[0]] = -x[:, idxs[0]]
		x[:, idxs[1]] = -x[:, idxs[1]]
		x[:, idxs[2]] = -x[:, idxs[2]]
		x[:, idxs[3]] = x[:, idxs[3]] + shift[3, :].T.astype(int)
	elif node == 4:
		#opt.nmiss[node] += 1
		x[:, idxs[0]] = -x[:, idxs[0]]
		x[:, idxs[1]] = -x[:, idxs[1]]
		x[:, idxs[2]] = -x[:, idxs[2]]
		x[:, idxs[3]] = -x[:, idxs[3]]
	# if node >= 3:
	# 	print('x', x)
	# 	print('-------------------------------')
	return x


def getX2D(output, datacat, width, height, outw, outh, isGauss, sigma, noise=False, node=0):
	global opt
	opt = get_opt()
	if datacat == 'chair' or datacat == 'bed':
		param_length = [5, 6, 12, 14]
	elif datacat == 'swivelchair' or datacat == 'sofa':
		param_length = [7, 8, 14, 16]
	elif datacat == 'table':
		param_length = [6, 7, 13, 15]
	elif datacat == 'car':
		param_length = [5, 6, 12, 14]#pca
		#param_length = [10, 11, 17, 19]

	alphaPred = np.array([output[0 : param_length[0]]])
	finvPred = output[param_length[0] : param_length[1]]
	sincosThetaPred = np.array([output[param_length[1] : param_length[2]]])
	sincosThetaPred[0, 2] = 0
	sincosThetaPred[0, 5] = 1
	tranPred = np.array([output[param_length[2] : param_length[3]]])
	stickStruct = getStickFigure(**{'class':datacat})
	im, x, coor3RT = visualize3Dpara(alphaPred.T, sincosThetaPred.T, tranPred.T, 1 / finvPred, 
						stickStruct['baseShape'], stickStruct['edgeAdj'], 
						**{'h':height, 'w':width})
	x[0, :] *= outw / width
	x[1, :] *= outh / height

	if noise:
		x = perturb(x, node)

	if isGauss:
		gauss_kernel = gaussian2D(6*sigma+1, sigma)
		x = draw_gaussian(x.T, sigma, gauss_kernel, stickStruct['np'], outh, outw)

	# for aa in xrange(10):
	# 	summ = 0
	# 	for bb in xrange(64):
	# 		for cc in xrange(64):
	# 			summ += x[bb, cc, aa]
	# 	print('summ', summ)
	summ = np.sum(np.sum(x, axis=0), axis=0)
	count = np.sum(summ == 0)
	if count >= 7:
		count = 7
	opt.nmiss[count] += 1
	# if node >= 3:
	# 	print(summ)
	# 	print('count', count)

	return x, coor3RT

def visualize3Dpara(alpha, sincosTheta, tran, f, baseShape, edgeAdj, **kwargs):
	para = {}
	para['h'] = 320
	para['w'] = 240
	para['lineWidth'] = 6
	para['addNode'] = True
	para['circSize'] = 8
	for key in kwargs.keys():
		para[key] = kwargs[key]

	theta = sctheta2theta(sincosTheta)
	x, coor3RT = alpha2x_proj(tran, alpha, theta, f, baseShape, **{'w':para['w'], 'h':para['h']})
	
	return None, x, coor3RT

def alpha2x_proj(tran, alpha, theta, f, baseShape, **kwargs):
	para = {}
	para['h'] = 64
	para['w'] = 64
	for key in kwargs.keys():
		para[key] = kwargs[key]
	h = para['h']
	h2 = h / 2.0
	w = para['w']
	w2 = w / 2.0

	nump = baseShape.shape[1]
	#nShape = alpha.shape[1]
	#x = np.zeros((2, nump, nShape))
	#for i in xrange(nShape):
	x = np.zeros((2, nump))
	Rx = np.array([[1,0,0], [0,math.cos(theta[0]),-math.sin(theta[0])], [0,math.sin(theta[0]),math.cos(theta[0])]])
	Ry = np.array([[math.cos(theta[1]),0,math.sin(theta[1])], [0,1,0], [-math.sin(theta[1]),0,math.cos(theta[1])]])
	Rz = np.array([[math.cos(theta[2]),-math.sin(theta[2]),0], [math.sin(theta[2]),math.cos(theta[2]),0], [0,0,1]])
	R = np.dot(Rx, Ry, Rz)
	alpha = alpha.T

	if opt.pca:
		inmean = opt.inmean
		coor3 = (np.sum(baseShape * alpha[np.newaxis, :], axis=2) + inmean) * 300.0
	else:
		coor3 = np.sum(baseShape * alpha[np.newaxis, :], axis=2)
	coor3RT = np.dot(R, coor3) + np.vstack((tran, np.array([0])))
	x = coor3RT[0:2, :] * (1.0 / (1.0 + coor3RT[2, :] / f)) + np.array([[w2], [h2]])

	return x, coor3RT

def sctheta2theta(sincosTheta):
	tmp = np.sqrt(sincosTheta[0:3]**2 + sincosTheta[3:6]**2)
	sincosTheta = sincosTheta / np.vstack((tmp, tmp))
	theta = np.arcsin(sincosTheta[0:3])
	mask = sincosTheta[3:6] < 0
	theta[mask] = math.pi - theta[mask]
	
	return theta

def getStickFigure(**kwargs):
	para = {}
	para['scale'] = []
	para['class'] = 'chair'
	para['indexPermute'] = True
	for key in kwargs.keys():
		para[key] = kwargs[key]

	stickStruct = {}
	stickStruct['nclass'] = 1

	if len(para['scale']) != 0 and para['scale'] == 64:
		stickStruct['scaleRange'] = [(9/2, 11/2)]
	elif len(para['scale']) != 0 and para['scale'] == 128:
		stickStruct['scaleRange'] = [(9/2, 11/2)]

	if para['class'] == 'chair':
		stickStruct['np'] = 10
		stickStruct['nbasis'] = 5
		stickStruct['edgeAdj'] = np.array([[1,5], [2,6], [3,7], [4,8], [5,6], [6,7], [7,8], [5,8], [8,9], [7,10], [9,10]]) - 1
		stickStruct['baseShape'] = np.zeros((3, stickStruct['np'], stickStruct['nbasis']))
		stickStruct['baseShape'][:,:,0] = np.array([[-1,-2,1], [1,-2,1], [1,-2,-1], [-1,-2,-1], [-1,0,1], 
													[1,0,1], [1,0,-1], [-1,0,-1], [-1,2,-1], [1,2,-1]]).T
		stickStruct['baseShape'][:,:,1] = np.array([[0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0], [0,0,0], 
													[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]).T
		stickStruct['baseShape'][:,:,2] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], 
													[0,0,0], [0,0,0], [0,0,0], [0,1,0], [0,1,0]]).T
		stickStruct['baseShape'][:,:,3] = np.array([[-1,0,0], [1,0,0], [1,0,0], [-1,0,0], [-1,0,0], 
													[1,0,0], [1,0,0], [-1,0,0], [-1,0,0], [1,0,0]]).T
		stickStruct['baseShape'][:,:,4] = np.array([[-1,0,1], [1,0,1], [1,0,-1], [-1,0,-1], [0,0,0], 
													[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]).T
		stickStruct['alphaRange'] = [(-1.4, 2), (-1.4, 2), (-0.3, 1), (0, 0.3)]
		stickStruct['alphaGSif'] = False
		stickStruct['scaleRange'] = [(30, 50)]
		stickStruct['thetaRange'] = [(1*math.pi, 1.15*math.pi), (0, 2*math.pi), (0, 0)]
		stickStruct['tranRange'] = [(-60, 60), (-80, 80)]
		stickStruct['fRange'] = [(120, 450)]
		stickStruct['h'] = 320
		stickStruct['w'] = 240 #320
		#stickStruct['indices'] = np.array([1,2,3,4,5,6,7,8,9,10])
		stickStruct['indices'] = np.array([0,1,2,3,4,5,6,7,8,9])
		stickStruct['shapecheckFunc'] = []
	elif para['class'] == 'wheelchair' or para['class'] == 'swivelchair':
		stickStruct['np'] = 13
		stickStruct['nbasis'] = 7
		stickStruct['edgeAdj'] = np.array([[1,6], [2,6], [3,6], [4,6], [5,6], [6,7], [8,9], 
											[9,10], [8,11], [10,11], [11,12], [12,13], [10,13]]) - 1
		stickStruct['baseShape'] = np.zeros((3, stickStruct['np'], stickStruct['nbasis']))
		tmpScale = 1.2
		stickStruct['baseShape'][:,:,0] = np.array([[0.3090*tmpScale,-1.5,0.9511*tmpScale], [-0.8090*tmpScale,-1.5,0.5878*tmpScale], 
													[-0.8090*tmpScale,-1.5,-0.5878*tmpScale], [0.3090*tmpScale,-1.5,-0.9511*tmpScale], 
													[1.0000*tmpScale,-1.5,0.0000*tmpScale], [0,-1.5,0], [0,0,0], [-1,0,1], [1,0,1],
													[1,0,-1], [-1,0,-1], [-1,1.5,-1], [1,1.5,-1]]).T
		stickStruct['baseShape'][:,:,1] = np.array([[0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0], [0,0,0], 
													[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]).T
		stickStruct['baseShape'][:,:,2] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], 
													[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,1,0], [0,1,0]]).T
		stickStruct['baseShape'][:,:,3] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], 
													[-1,0,0], [1,0,0], [1,0,0], [-1,0,0], [-1,0,0], [1,0,0]]).T
		stickStruct['baseShape'][:,:,4] = np.array([[0.3090,0,0.9511], [-0.8090,0,0.5878], [-0.8090,0,-0.5878], 
													[0.3090,0,-0.9511], [1.0000,0,0.0000], [0,0,0], [0,0,0], 
													[0,0,0], [0,0,0], [0,0,0], [0,0,0],	[0,0,0], [0,0,0]]).T
		stickStruct['baseShape'][:,:,5] = np.array([[-0.9511,0,0.3090], [-0.5878,0,-0.8090], [0.5878,0,-0.8090], 
													[0.9511,0,0.3090], [0.0000,0,1.0000], [0,0,0], [0,0,0],
													[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]).T
		stickStruct['baseShape'][:,:,6] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,1,0], [0,0,0],
													[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]).T
		stickStruct['alphaRange'] = [(-0.5, 0.5), (-1, 2.5), (-0.5, 1), (-0.5, 0.8), (-0.3, 0.6), (0, 0.6)]
		stickStruct['alphaGSif'] = False
		stickStruct['scaleRange'] = [(35, 85)]
		stickStruct['thetaRange'] = [(1*math.pi, 1.15*math.pi), (0, 2*math.pi), (0, 0)]
		stickStruct['tranRange'] = [(-60, 60), (-80, 80)]
		stickStruct['fRange'] = [(120, 450)]
		stickStruct['h'] = 320
		stickStruct['w'] = 240
		#stickStruct['indices'] = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
		stickStruct['indices'] = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
		stickStruct['shapecheckFunc'] = []
	elif para['class'] == 'table':
		stickStruct['np'] = 8
		stickStruct['nbasis'] = 6
		stickStruct['edgeAdj'] = np.array([[1,2], [2,4], [1,3], [3,4], [1,5], [2,6], [3,7], [4,8]]) - 1
		stickStruct['baseShape'] = np.zeros((3, stickStruct['np'], stickStruct['nbasis']))
		stickStruct['baseShape'][:,:,0] = np.array([[-1,1,-1], [1,1,-1], [-1,1,1], [1,1,1], 
													[-1,-1,-1],	[1,-1,-1], [-1,-1,1], [1,-1,1]]).T
		stickStruct['baseShape'][:,:,1] = np.array([[0,1,0], [0,1,0], [0,1,0], [0,1,0], 
													[0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0]]).T
		stickStruct['baseShape'][:,:,2] = np.array([[-1,0,0], [1,0,0], [-1,0,0], [1,0,0], 
													[-1,0,0], [1,0,0], [-1,0,0], [1,0,0]]).T
		stickStruct['baseShape'][:,:,3] = np.array([[0,0,-1], [0,0,-1], [0,0,1], [0,0,1], 
													[0,0,-1], [0,0,-1], [0,0,1], [0,0,1]]).T
		stickStruct['baseShape'][:,:,4] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], 
													[1,0,0], [-1,0,0], [1,0,0], [-1,0,0]]).T
		stickStruct['baseShape'][:,:,5] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], 
													[0,0,-1], [0,0,-1], [0,0,1], [0,0,1]]).T
		stickStruct['alphaRange'] = [(-0.3, 0.3), (-0.6, 2.5), (-0.3, 1), (0, 1), (0, 0.2)]
		stickStruct['alphaGSif'] = False
		stickStruct['scaleRange'] = [(30, 42)]
		stickStruct['thetaRange'] = [(1*math.pi, 1.3*math.pi), (-0.23*math.pi, 0.23*math.pi), (0, 0)]
		stickStruct['tranRange'] = [(-80, 80), (-60, 60)]
		stickStruct['fRange'] = [(120, 450)]
		stickStruct['h'] = 320 # 240
		stickStruct['w'] = 240
		#stickStruct['indices'] = np.array([1,2,3,4,5,6,7,8])
		stickStruct['indices'] = np.array([0,1,2,3,4,5,6,7])
		stickStruct['shapecheckFunc'] = []
	elif para['class'] == 'bed':
		stickStruct['np'] = 10
		stickStruct['nbasis'] = 5
		stickStruct['edgeAdj'] = np.array([[1,5], [2,6], [3,7], [4,8], [3,4], [5,6], 
											[6,7], [7,8], [5,8], [8,9], [7,10], [9,10]]) - 1
		stickStruct['baseShape'] = np.zeros((3, stickStruct['np'], stickStruct['nbasis']))
		stickStruct['baseShape'][:,:,0] = np.array([[-1.00,-1.00,2.00], [1.00,-1.00,2.00], [1.00,-1.00,-2.00], [-1.00,-1.00,-2.00], 
													[-1.00,0.00,2.00], [1.00,0.00,2.00], [1.00,0.00,-2.00], [-1.00, 0.00,-2.00], 
        											[-1.00,1.00,-2.00], [1.00,1.00,-2.00]]).T
		stickStruct['baseShape'][:,:,1] = np.array([[0,0,1], [0,0,1], [0,0,-1], [0,0,-1], [0,0,1], 
													[0,0,1], [0,0,-1], [0,0,-1], [0,0,-1], [0,0,-1]]).T
		stickStruct['baseShape'][:,:,2] = np.array([[-1,0,0], [1,0,0], [1,0,0], [-1,0,0], [-1,0,0], 
													[1,0,0], [1,0,0], [-1,0,0], [-1,0,0], [1,0,0]]).T
		stickStruct['baseShape'][:,:,3] = np.array([[0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0], [0,0,0], 
													[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]).T
		stickStruct['baseShape'][:,:,4] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], 
													[0,0,0], [0,0,0], [0,0,0], [0,1,0], [0,1,0]]).T
		stickStruct['alphaRange'] = [(-1.2, 0.5), (-0.5, 1.5), (-0.5, 0.5), (-0.5, 1)]
		stickStruct['alphaGSif'] = False
		stickStruct['scaleRange'] = [(30, 42)]
		stickStruct['thetaRange'] = [(1*math.pi, 1.25*math.pi), (-0.75*math.pi, 0.75*math.pi), (0, 0)]
		stickStruct['tranRange'] = [(-80, 80), (-60, 60)]
		stickStruct['fRange'] = [(200, 450)]
		stickStruct['h'] = 320 #240
		stickStruct['w'] = 240
		#stickStruct['indices'] = np.array([2,7,6,1,4,9,8,3,5,10])
		stickStruct['indices'] = np.array([1,6,5,0,3,8,7,2,4,9])
		stickStruct['shapecheckFunc'] = []
	elif para['class'] == 'sofa':
		stickStruct['np'] = 14
		stickStruct['nbasis'] = 7
		stickStruct['edgeAdj'] = np.array([[1,5], [2,6], [3,7], [4,8], [5,6], [6,7], [7,8], [5,8], 
									[8,9], [7,10], [9,10], [11,12], [12,5], [13,14], [14,6]]) - 1
		stickStruct['baseShape'] = np.zeros((3, stickStruct['np'], stickStruct['nbasis']))
		stickStruct['baseShape'][:,:,0] = np.array([[-2.00,-2.00,1.00], [2.00,-2.00,1.00], [2.00,-2.00,-1.00], 
													[-2.00,-2.00,-1.00], [-2.00,0.00,1.00], [2.00,0.00,1.00], 
													[2.00,0.00,-1.00], [-2.00,0.00,-1.00], [-2.00,2.00,-1.00],
													[2.00,2.00,-1.00], [-2.00,1.00,-1.00], [-2.00,1.00,1.00], 
													[2.00,1.00,-1.00], [2.00,1.00,1.00]]).T
		stickStruct['baseShape'][:,:,1] = np.array([[0,0,1], [0,0,1], [0,0,-1], [0,0,-1], [0,0,1], [0,0,1], [0,0,-1], 
													[0,0,-1], [0,0,-1], [0,0,-1], [0,0,-1], [0,0,1], [0,0,-1], [0,0,1]]).T
		stickStruct['baseShape'][:,:,2] = np.array([[-1,0,0], [1,0,0], [1,0,0], [-1,0,0], [-1,0,0], [1,0,0], [1,0,0], 
													[-1,0,0], [-1,0,0], [1,0,0], [-1,0,0], [-1,0,0], [1,0,0], [1,0,0]]).T
		stickStruct['baseShape'][:,:,3] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], 
													[0,0,0], [0,1,0], [0,1,0], [0,0.5,0], [0,0.5,0], [0,0.5,0], [0,0.5,0]]).T
		stickStruct['baseShape'][:,:,4] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], 
													[0,0,0], [0,0,0], [0,0,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0]]).T
		stickStruct['baseShape'][:,:,5] = np.array([[0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0], [0,0,0], [0,0,0], [0,0,0],
													[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]).T
		stickStruct['baseShape'][:,:,6] = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0],
													[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,-1], [0,0,0], [0,0,-1]]).T
		stickStruct['alphaRange'] = [(-0.5, 0.5), (0, 2), (-0.5, 1), (-1, 2), (-1, 1), (0, 2)]
		stickStruct['alphaGSif'] = False
		stickStruct['scaleRange'] = [(30, 42)]
		stickStruct['thetaRange'] = [(1*math.pi, 1.15*math.pi), (-0.5*math.pi, 0.5*math.pi), (0, 0)]
		stickStruct['tranRange'] = [(-80, 80), (-60, 60)]
		stickStruct['fRange'] = [(200, 450)]
		stickStruct['h'] = 320 #240
		stickStruct['w'] = 240
		#stickStruct['indices'] = np.array([2,9,8,1,4,11,10,3,7,14,5,6,12,13])
		stickStruct['indices'] = np.array([1,8,7,0,3,10,9,2,6,13,4,5,11,12])
		stickStruct['shapecheckFunc'] = []
	elif para['class'] == 'car':
		stickStruct['np'] = opt.nKeypoints
		stickStruct['nbasis'] = opt.OutChannels - 9 #pca 5  kmeans 10
		stickStruct['edgeAdj'] = np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,1],
											[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],[17,18],[18,11],
											[1,11],[2,12],[3,13],[4,14],[5,15],[6,16],[7,17],[8,18]]) - 1
											# [9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,1],
											# [19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,26],[26,27],
											# [27,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,19],
											# [1,19],[2,20],[3,21],[4,22],[5,23],[6,24],[7,25],[8,26]]) - 1
		stickStruct['baseShape'] = np.zeros((3, stickStruct['np'], stickStruct['nbasis']))
		c = scipy.io.loadmat(os.path.join(opt.dataDir, 'car', 'centers.mat'))
		centers = c['centers']
		for i in xrange(stickStruct['nbasis']):
			for j in xrange(opt.nKeypoints):
				for k in xrange(3):
					stickStruct['baseShape'][k,j,i] = centers[i, j + opt.nKeypoints*k]


		stickStruct['alphaRange'] = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), 
									(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), 
									(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), 
									(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
		stickStruct['alphaGSif'] = False
		stickStruct['scaleRange'] = [(30, 42)]
		stickStruct['thetaRange'] = [(1*math.pi, 1.15*math.pi), (-0.5*math.pi, 0.5*math.pi), (0, 0)]
		stickStruct['tranRange'] = [(-80, 80), (-60, 60)]
		stickStruct['fRange'] = [(200, 450)]
		stickStruct['h'] = 320 #240
		stickStruct['w'] = 240
		#stickStruct['indices'] = np.array([2,9,8,1,4,11,10,3,7,14,5,6,12,13])
		stickStruct['indices'] = np.array(range(opt.nKeypoints))
		stickStruct['shapecheckFunc'] = []

	if para['indexPermute']:
		forwardidx = stickStruct['indices']
		revertidx = np.zeros(forwardidx.shape[0])
		revertidx[forwardidx] = np.array(range(forwardidx.shape[0]))
		#stickStruct['baseShape'] = stickStruct['baseShape'][:, revertidx, :]
		tmp = np.zeros(stickStruct['baseShape'].shape)
		for i in xrange(forwardidx.shape[0]):
			tmp[:,i,:] = copy.deepcopy(stickStruct['baseShape'][:,int(revertidx[i]),:])
		stickStruct['baseShape'] = copy.deepcopy(tmp)
		for i in xrange(stickStruct['edgeAdj'].shape[0]):
			stickStruct['edgeAdj'][i][0] = forwardidx[stickStruct['edgeAdj'][i][0]]
			stickStruct['edgeAdj'][i][1] = forwardidx[stickStruct['edgeAdj'][i][1]]

	if len(para['scale']) != 0 and para['scale'] == 64:
		stickStruct['scaleRange'] = [(9/2, 11/2)]
	elif len(para['scale']) != 0 and para['scale'] == 128:
		stickStruct['scaleRange'] = [(9/2, 11/2)]

	return stickStruct

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

def draw_gaussian(label, sigma, gauss_kernel, num_kp, h, w):
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

def plot_3Dshape():
	stickStruct = getStickFigure(**{'class':'swivelchair'})
	base1 = stickStruct['baseShape'][:,:,0]# + stickStruct['baseShape'][:,:,6]
	verts = []
	for i in xrange(base1.shape[1]):
		verts.append(tuple(base1[:,i]))
	faces = []
	edges = stickStruct['edgeAdj']
	for i in xrange(edges.shape[0]):
		faces.append(list(edges[i,:]))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	#verts = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]
	#faces = [[0, 1, 2, 3], [4, 5, 6, 7]]#, [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
	poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
	print(poly3d)
	x, y, z = zip(*verts)
	ax.scatter(x, y, z)
	#ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidths=1, alpha=0.3))
	ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=1, linestyles=':'))
	ax.set_xlabel('X')
	ax.set_xlim3d(-3.5, 3.5)
	ax.set_ylabel('Y')
	ax.set_ylim3d(-3.5, 3.5)
	ax.set_zlabel('Z')
	ax.set_zlim3d(-3.5, 3.5)
	plt.show()

def plot_2Dshpae(datacat, x, w, h):
	stickStruct = getStickFigure(**{'class':datacat})
	edges = stickStruct['edgeAdj']
	num_kp = stickStruct['np']
	f = plt.figure(figsize=(4,4))
	ax = plt.subplot(2,num_kp,1)
	ax.set_xlim(left=0, right=w)
	ax.set_ylim(bottom=h, top=0)
	# line1 = [(1, 1), (5, 5)]
	# line2 = [(11, 9), (8, 8)]
	# (line1_xs, line1_ys) = zip(*line1)
	# (line2_xs, line2_ys) = zip(*line2)
	# ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))
	# ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='red'))
	ax.xaxis.tick_top()
	ax.set_aspect(1)

	for i in xrange(edges.shape[0]):
		plt.plot([x[0][edges[i][0]], x[0][edges[i][1]]], [x[1][edges[i][0]], x[1][edges[i][1]]], 
				'ko-', label='line 1', linewidth=2)
	#plt.axis([0, w, h, 0])
	sigma = 1
	gauss_kernel = gaussian2D(6*sigma+1, sigma)
	heatmap = draw_gaussian(x.T, sigma, gauss_kernel, num_kp, h, w)
	for i in xrange(num_kp):
		ax = plt.subplot(2, num_kp, i+1+num_kp)
		#ax.axis('off')
		plt.imshow(heatmap[:,:,i])
	plt.show()

def plot_curve(loss, acc):
	plt.subplot(1, 2, 1)
	plt.plot(loss['train'], label='train')
	plt.plot(loss['val'], label='val')
	plt.title('Loss history')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	plt.subplot(1, 2, 2)
	plt.plot(acc['train'], label='train')
	plt.plot(acc['val'], label='val')
	plt.title('Accuracy history')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')

	plt.savefig('out/result.png')


# if __name__ == '__main__':
#  	gt_tmp = scipy.io.loadmat('chair_3dinn.mat')
#  	gt = gt_tmp['outputs']	# (194, 14)
#  	datacat = 'chair'
#  	#for i in xrange(gt.shape[0]):
#  	x, coor3RT = getX2D(gt[0], datacat, 320, 320, 64, 64, False, 1)
#  	print(x)
#  	plot_2Dshpae(datacat, x, 64, 64)