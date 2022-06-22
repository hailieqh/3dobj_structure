from __future__ import print_function, division
import os
import math
import copy
import scipy.io
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import collections
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from torchvision import transforms, utils
from opts import *

class ThreeDNet(nn.Module):
	def __init__(self, InChannels, OutChannels):
		super(ThreeDNet, self).__init__()
		global opt
		opt = get_opt()
		self.InChannels = InChannels
		self.OutChannels = OutChannels
		self.fc1 = nn.Linear(InChannels, 2048)
		self.fc2 = nn.Linear(2048, 512)
		self.fc3 = nn.Linear(512, 128)
		self.fc4 = nn.Linear(128, OutChannels)
		self.relu = nn.ReLU()
		self.drop = nn.Dropout(p=opt.drop)

	def forward(self, x):
		x = x.view(-1, self.InChannels)
		x = self.drop(self.relu(self.fc1(x)))
		x = self.drop(self.relu(self.fc2(x)))
		x = self.drop(self.relu(self.fc3(x)))
		out = self.fc4(x)
		return out

#class Projection(Function):
#	def forward(self, input):

def postprocess(outputs, datacat):
	outputs_post = Variable(copy.deepcopy(outputs.data))
	if datacat == 'chair' or datacat == 'bed':
		param_length = [5, 6, 12, 14]
	elif datacat == 'swivelchair' or datacat == 'sofa':
		param_length = [7, 8, 14, 16]
	elif datacat == 'table':
		param_length = [6, 7, 13, 15]
	elif datacat == 'car':
		param_length = [5, 6, 12, 14]#pca
		#param_length = [10, 11, 17, 19]
	batch = outputs_post.size(0)
	a = param_length[0] - 1
	b = param_length[1]
	for i in xrange(batch):
		outputs_post.data[i][0] = min(outputs_post.data[i][0], 50)
		outputs_post.data[i][a] = max(outputs_post.data[i][a], 0)
		for j in xrange(6):
			outputs_post.data[i][j+b] = max(outputs_post.data[i][j+b], -1)
			outputs_post.data[i][j+b] = min(outputs_post.data[i][j+b], 1)
	return outputs_post

def projection(outputs, datacat, width, height, outw, outh, isGauss, sigma):
	batch = outputs.size(0)
	if not isGauss:
		for i in xrange(batch):
			output_unnorm_0 = unnorm_tensor(outputs[i])
			if i == 0:
				output_unnorm = torch.unsqueeze(output_unnorm_0, 0)
			else:
				output_unnorm = torch.cat((output_unnorm, torch.unsqueeze(output_unnorm_0, 0)), 0)
		return output_unnorm
	else:
		for i in xrange(batch):
			output_unnorm_0 = unnorm_tensor(outputs[i])
			x_0, coor3RT_0 = getX2D_tensor(output_unnorm_0, datacat, width, height, outw, outh, isGauss, sigma)
			if i == 0:
				x = torch.unsqueeze(x_0, 0)
				coor3RT = torch.unsqueeze(coor3RT_0, 0)
				output_unnorm = torch.unsqueeze(output_unnorm_0, 0)
			else:
				x = torch.cat((x, torch.unsqueeze(x_0, 0)), 0)
				coor3RT = torch.cat((coor3RT, torch.unsqueeze(coor3RT_0, 0)), 0)
				output_unnorm = torch.cat((output_unnorm, torch.unsqueeze(output_unnorm_0, 0)), 0)
		return x, coor3RT, output_unnorm
	
		
def unnorm_tensor(output):
	global opt
	opt = get_opt()
	if opt.dataset == 'chair':
		mul_mask = torch.FloatTensor([20, 170, 170, 65, 15, 1/350-1/450, 2, 2, 2, 2, 2, 2, 120, 160])
		add_mask = torch.FloatTensor([30, -70, -70, -25, 0, 1/450, -1, -1, -1, -1, -1, -1, -60, -80])
	elif opt.dataset == 'sofa':
		mul_mask = torch.FloatTensor([12, 42, 84, 63, 126, 84, 64, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		add_mask = torch.FloatTensor([30, -11, 0, -21, -62, -42, 0, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
	elif opt.dataset == 'swivelchair':
		mul_mask = torch.FloatTensor([50, 105, 257.5, 107.5, 107.5, 76.5, 51, 1/350-1/450, 2, 2, 2, 2, 2, 2, 120, 160])
		add_mask = torch.FloatTensor([25, -42.5, -95, -42.5, -42.5, -25.5, 0, 1/450, -1, -1, -1, -1, -1, -1, -60, -80])
	elif opt.dataset == 'table':
		# mul_mask = torch.FloatTensor([40, 100, 100, 10, 50, 50, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		# add_mask = torch.FloatTensor([30, -50, 0, 0, -25, -25, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
		# mul_mask = torch.FloatTensor([40, 50, 124, 70, 40, 40, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		# add_mask = torch.FloatTensor([30, -25, -24, -10, -20, -20, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
		# mul_mask = torch.FloatTensor([30, 50, 124, 42, 20, 20, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		# add_mask = torch.FloatTensor([40, -25, -24, -12, -10, -10, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
		mul_mask = torch.FloatTensor([30, 50, 124, 100, 40, 40, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		add_mask = torch.FloatTensor([30, -25, -24, -10, -20, -20, 1/450, -1, -1, -1, -1, -1, -1, -80, -60])
	elif opt.dataset == 'car':
		#kmeans
		# mul_mask = torch.FloatTensor([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		# add_mask = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/650, -1, -1, -1, -1, -1, -1, -80, -60])
		#pca
		mul_mask = torch.FloatTensor([1, 1, 1, 1, 1, 1/350-1/450, 2, 2, 2, 2, 2, 2, 160, 120])
		add_mask = torch.FloatTensor([0, 0, 0, 0, 0, 1/650, -1, -1, -1, -1, -1, -1, -80, -60])
	output = torch.mul(output, Variable(mul_mask).cuda())
	output = torch.add(output, Variable(add_mask).cuda())
	return output

def getX2D_tensor(output, datacat, width, height, outw, outh, isGauss, sigma):
	if datacat == 'chair' or datacat == 'bed':
		param_length = [5, 6, 12, 14]
	elif datacat == 'swivelchair' or datacat == 'sofa':
		param_length = [7, 8, 14, 16]
	elif datacat == 'table':
		param_length = [6, 7, 13, 15]
	elif datacat == 'car':
		param_length = [5, 6, 12, 14]#pca
		#param_length = [10, 11, 17, 19]

	length = output.size()
	alpha_mask = torch.zeros(length).byte()
	finv_mask = torch.zeros(length).byte()
	sincosTheta_mask = torch.zeros(length).byte()
	tran_mask = torch.zeros(length).byte()
	for i in xrange(0, param_length[0]):
		alpha_mask[i] = 1
	for i in xrange(param_length[0], param_length[1]):
		finv_mask[i] = 1
	for i in xrange(param_length[1], param_length[2]):
		sincosTheta_mask[i] = 1
	for i in xrange(param_length[2], param_length[3]):
		tran_mask[i] = 1

	alphaPred = torch.masked_select(output, Variable(alpha_mask).cuda())
	finvPred = torch.masked_select(output, Variable(finv_mask).cuda())
	sincosThetaPred = torch.masked_select(output, Variable(sincosTheta_mask).cuda())
	sincosTheta_adj = torch.ones(sincosThetaPred.size())
	sincosTheta_adj[2] = 0
	sincosTheta_adj[5] = torch.reciprocal(sincosThetaPred[5]).data[0]
	sincosThetaPred = torch.mul(sincosThetaPred, Variable(sincosTheta_adj).cuda())
	tranPred = torch.masked_select(output, Variable(tran_mask).cuda())
	alphaPred = torch.unsqueeze(alphaPred, 0)
	sincosThetaPred = torch.unsqueeze(sincosThetaPred, 0)
	tranPred = torch.unsqueeze(tranPred, 0)
	stickStruct = getStickFigure_tensor(**{'class':datacat})
	im, x, coor3RT = visualize3Dpara_tensor(torch.t(alphaPred), torch.t(sincosThetaPred), 
						torch.t(tranPred), 1 / (finvPred + 0.0000001), 
						stickStruct['baseShape'], stickStruct['edgeAdj'], 
						**{'h':height, 'w':width})
	scale = torch.FloatTensor([[outw / width], [outh / height]])
	x = torch.mul(x, Variable(scale).cuda())

	if isGauss:
		gauss_kernel = gaussian2D_tensor(6*sigma+1, sigma)
		x = draw_gaussian_tensor(torch.t(x), sigma, gauss_kernel, stickStruct['np'], outh, outw)

	return x, coor3RT

def visualize3Dpara_tensor(alpha, sincosTheta, tran, f, baseShape, edgeAdj, **kwargs):
	para = {}
	para['h'] = 320
	para['w'] = 240
	para['lineWidth'] = 6
	para['addNode'] = True
	para['circSize'] = 8
	for key in kwargs.keys():
		para[key] = kwargs[key]

	theta = sctheta2theta_tensor(sincosTheta)
	x, coor3RT = alpha2x_proj_tensor(tran, alpha, theta, f, baseShape, **{'w':para['w'], 'h':para['h']})
	
	return None, x, coor3RT

def alpha2x_proj_tensor(tran, alpha, theta, f, baseShape, **kwargs):
	para = {}
	para['h'] = 64
	para['w'] = 64
	for key in kwargs.keys():
		para[key] = kwargs[key]
	h = para['h']
	h2 = h / 2.0
	w = para['w']
	w2 = w / 2.0

	nump = baseShape.size(1)
	x = torch.zeros((2, nump))
	theta_mask_0 = torch.ByteTensor([1, 0, 0])
	theta_mask_1 = torch.ByteTensor([0, 1, 0])
	theta_mask_2 = torch.ByteTensor([0, 0, 1])
	theta_0 = torch.masked_select(torch.t(theta), Variable(theta_mask_0).cuda())
	theta_1 = torch.masked_select(torch.t(theta), Variable(theta_mask_1).cuda())
	theta_2 = torch.masked_select(torch.t(theta), Variable(theta_mask_2).cuda())
	# RX
	x_0 = Variable(torch.FloatTensor([[1], [0], [0]])).cuda()
	x_1 = torch.stack((Variable(torch.FloatTensor([0])).cuda(), torch.cos(theta_0), -torch.sin(theta_0)), 0)
	x_2 = torch.stack((Variable(torch.FloatTensor([0])).cuda(), torch.sin(theta_0), torch.cos(theta_0)), 0)
	Rx = torch.cat((torch.t(x_0), torch.t(x_1), torch.t(x_2)), 0)
	# RY
	y_0 = torch.stack((torch.cos(theta_1), Variable(torch.FloatTensor([0])).cuda(), torch.sin(theta_1)), 0)
	y_1 = Variable(torch.FloatTensor([[0], [1], [0]])).cuda()
	y_2 = torch.stack((-torch.sin(theta_1), Variable(torch.FloatTensor([0])).cuda(), torch.cos(theta_1)), 0)
	Ry = torch.cat((torch.t(y_0), torch.t(y_1), torch.t(y_2)), 0)
	# RZ
	z_0 = torch.stack((torch.cos(theta_2), -torch.sin(theta_2), Variable(torch.FloatTensor([0])).cuda()), 0)
	z_1 = torch.stack((torch.sin(theta_2), torch.cos(theta_2), Variable(torch.FloatTensor([0])).cuda()), 0)
	z_2 = Variable(torch.FloatTensor([[0], [0], [1]])).cuda()
	Rz = torch.cat((torch.t(z_0), torch.t(z_1), torch.t(z_2)), 0)
	R = torch.mm(torch.mm(Rx, Ry), Rz)

	alpha = torch.unsqueeze(torch.t(alpha), 0)
	if opt.pca:
		inmean = Variable(torch.from_numpy(opt.inmean).float()).cuda()
		coor3 = (torch.sum(Variable(baseShape).cuda() * alpha, 2) + inmean) * 300.0
	else:
		coor3 = torch.sum(Variable(baseShape).cuda() * alpha, 2)
	coor3RT = torch.mm(R, coor3) + torch.cat((tran, Variable(torch.zeros(1)).cuda()), 0)

	indice_01 = torch.LongTensor([0, 1])
	indice_2 = torch.LongTensor([2])
	row_01 = torch.index_select(coor3RT, 0, Variable(indice_01).cuda())
	row_2 = torch.index_select(coor3RT, 0, Variable(indice_2).cuda())
	x = row_01 * (1.0 / (1.0 + row_2 / (f  + 0.0000001))) + Variable(torch.FloatTensor([[w2], [h2]])).cuda()

	return x, coor3RT

def sctheta2theta_tensor(sincosTheta):
	length = sincosTheta.size(0)
	mask_0 = torch.zeros(length).byte()
	mask_1 = torch.zeros(length).byte()
	mask_0[0] = mask_0[1] = mask_0[2] = 1
	mask_1[3] = mask_1[4] = mask_1[5] = 1
	sincosTheta_0 = torch.masked_select(torch.t(sincosTheta), Variable(mask_0).cuda())
	sincosTheta_1 = torch.masked_select(torch.t(sincosTheta), Variable(mask_1).cuda())
	tmp = torch.sqrt(torch.mul(sincosTheta_0, sincosTheta_0) +
					torch.mul(sincosTheta_1, sincosTheta_1))
	tmp = torch.cat((tmp, tmp), 0)
	sincosTheta = torch.t(sincosTheta) / (tmp + 0.0000001)
	theta = torch.asin(torch.masked_select(sincosTheta, Variable(mask_0).cuda()))
	mask = torch.lt(torch.masked_select(sincosTheta, Variable(mask_1).cuda()), 0)
	theta[mask] = math.pi - theta[mask]
	theta = torch.unsqueeze(theta, 1)
	
	return theta

def getStickFigure_tensor(**kwargs):
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

	stickStruct['baseShape'] = torch.from_numpy(stickStruct['baseShape']).float()
	stickStruct['edgeAdj'] = torch.from_numpy(stickStruct['edgeAdj']).long()

	return stickStruct

def gaussian2D_tensor(size, sigma):
	sigma_2 = 2 * sigma * sigma
	r_size = (size - 1)//2
	gk = torch.zeros((size, size))
	for i in range(-r_size, r_size + 1):
		h = i + r_size
		for j in range(-r_size, r_size + 1):
			w = j + r_size
			v = np.exp(-(i*i + j*j) / sigma_2)
			gk[h, w] = v
			if i*i + j*j > r_size*r_size:
				gk[h, w] = 0
	return gk

def draw_gaussian_tensor(label, sigma, gauss_kernel, num_kp, h, w):
	out = torch.zeros((h, w, num_kp)) # (64, 64, 10)
	for i in xrange(num_kp):
		x, y = label.data[i]
		if x < 0 or x >= 64 or y < 0 or y >= 64:
			continue
		x, y = int(x), int(y)
		r_size = 3 * sigma
		tmp = torch.zeros((64 + 2 * r_size, 64 + 2 * r_size))
		tmp[y : y+2*r_size+1, x : x+2*r_size+1] = gauss_kernel
		out[:, :, i] = tmp[r_size : 64+r_size, r_size : 64+r_size]
	return out

def plot_2Dshpae(datacat, x, w, h, idx):
	stickStruct = getStickFigure_tensor(**{'class':datacat})
	edges = stickStruct['edgeAdj']
	num_kp = stickStruct['np']
	f = plt.figure(figsize=(4,4))
	ax = plt.subplot(2,1,1)
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

	for i in xrange(edges.size(0)):
		plt.plot([x[0][edges[i][0]].data[0], x[0][edges[i][1]].data[0]], [x[1][edges[i][0]].data[0], x[1][edges[i][1]].data[0]], 
				'ko-', label='line 1', linewidth=2)
	#plt.axis([0, w, h, 0])
	sigma = 1
	# gauss_kernel = gaussian2D_tensor(6*sigma+1, sigma)
	# heatmap = draw_gaussian_tensor(torch.t(x), sigma, gauss_kernel, num_kp, h, w)
	# for i in xrange(num_kp):
	# 	ax = plt.subplot(2, num_kp, i+1+num_kp)
	# 	#ax.axis('off')
	# 	plt.imshow(heatmap[:,:,i])
	plt.savefig('out/result_{}.png'.format(idx))

# if __name__ == '__main__':
#  	gt_tmp = scipy.io.loadmat('chair_gt_val.mat')
#  	gt = gt_tmp['outputs']	# (194, 14)
#  	datacat = 'chair'
#  	#for i in xrange(gt.shape[0]):
#  	gt = torch.from_numpy(gt).float()
#  	#for i in xrange(100):
#  	outputs = Variable(torch.cat((torch.unsqueeze(gt[0], 0), torch.unsqueeze(gt[1], 0)), 0))
#  	x, coor3RT = projection(outputs, datacat, 320, 320, 64, 64, False, 1)
#  	print(x)
#  	print(coor3RT)
# 	#x, coor3RT = projection(Variable(torch.unsqueeze(gt[i], 0)), datacat, 320, 320, 64, 64, False, 1)
# 		#plot_2Dshpae(datacat, x, 64, 64, i)