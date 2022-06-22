from __future__ import print_function, division
import os
import shutil
import time
import datetime
import matplotlib
matplotlib.use('Agg')

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import numpy as np

from opts import *
from model import *
from mydata import *
from utils import *

def main():
	# create model
	model = HGNet()

	# define loss function (criterion) and optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.LR,
				alpha=opt.alpha, eps=opt.epsilon,
				weight_decay=opt.weightDecay, momentum=opt.momentum)

	# use cuda or not
	use_gpu = (opt.GPU != -1) and torch.cuda.is_available()
	if use_gpu == False:
		print('No GPU......')
	else:
		torch.cuda.set_device(opt.GPU)
		model = model.cuda()
		criterion = criterion.cuda()

	# load data
	data_transform = {}
	data_transform['train'] = transforms.Compose([
							RandomRColor(),
							#RandomCut(30), 
							PadSquare(), RandomRotate(),
							RandomHorizontalFlip(), 
							Rescale((256,256)), 
							ToTensor()])
	data_transform['val'] = transforms.Compose([
							PadSquare(),
							Rescale((256,256)),
							ToTensor()])
	kp_datasets = {x: KPDataset('coords.mat', opt.dataDir, transform=data_transform[x], 
						datacat=opt.dataset, phase=x) for x in ['train', 'val']}
	batch = {'train': opt.trainBatch, 'val': opt.validBatch}
	data_loaders = {x: DataLoader(kp_datasets[x], batch_size=batch[x], shuffle=True, drop_last=True)#, num_workers=batch[x])
						for x in ['train', 'val']}
	data_sizes = {x: (len(kp_datasets[x]) - len(kp_datasets[x]) % batch[x]) for x in ['train', 'val']}

	# testing
	if opt.test:
		demo(data_loaders, data_sizes, model, criterion, optimizer, use_gpu)
		return

	# training
	opt.PCK = {x: [] for x in ['train', 'val']}
	opt.PCP = {x: [] for x in ['train', 'val']}
	opt.AE = {x: [] for x in ['train', 'val']}
	opt.loss = {x: [] for x in ['train', 'val']}
	#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1-opt.LRdecay)
	train_val(data_loaders, data_sizes, batch, model, criterion, optimizer, use_gpu)
	#torch.save(best_model.state_dict(), '{}/best_model.pth'.format(opt.expDir))
	torch.save(opt.PCK, 'PCK.pth')
	torch.save(opt.PCP, 'PCP.pth')
	torch.save(opt.AE, 'AE.pth')
	torch.save(opt.loss, 'loss.pth')

def train_val(data_loaders, data_sizes, batch, model, criterion, optimizer, use_gpu):
	since = time.time()
	stdd = {}
	i_std = 0
	best_model = model
	best_acc_pck = 0.0
	best_acc_pcp = 0.0
	best_ae = 10.0

	best_wrist = 0.0
	best_elbow = 0.0

	for epoch in xrange(opt.nEpochs):
		print('Epoch {}/{}'.format(epoch, opt.nEpochs - 1))
		print('-' * 30)
		# if epoch == 1:
		# 	return
		# each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				#lr_scheduler.step()
				model.train(True) # set model to training mode
			else:
				model.train(False) # set model to evaluation mode

			running_loss = 0.0
			running_acc_pck = 0.0
			running_acc_pcp = 0.0
			running_ae = 0.0

			running_wrist = 0.0
			running_elbow = 0.0

			# iterate over data
			for i_batch, data_tmp in enumerate(data_loaders[phase]):
				# get the inputs
				data, std = data_tmp
				std = std.float()
				if epoch == 1:
					stdd[i_std] = std
					i_std += 1
				#print(std)
				inputs, labels = data['image'].float(), data['label'].float()

				# wrap them in Variable
				if use_gpu:
					inputs, labels = inputs.cuda(), labels.cuda()
				inputs, labels = Variable(inputs), Variable(labels)

				# forward
				outputs = model(inputs)
				loss = {}
				for i in xrange(opt.nStack):
					loss[i] = criterion(outputs[i], labels)
				#loss = criterion(outputs, labels)

				# backward & optimize, only if in training phase
				if phase == 'train':
					optimizer.zero_grad()
					sumloss = loss[0]
					for i in xrange(1, opt.nStack):
						sumloss += loss[i]
					sumloss.backward()
					#loss.backward()
					optimizer.step()
					outputs = outputs[opt.nStack - 1]
				else:
					# validation: get flipped outputs
					inputs_flip = flip(inputs)
					inputs_flip = inputs_flip.float()
					if use_gpu:
						inputs_flip = inputs_flip.cuda()
					outputs_flip = model(Variable(inputs_flip))
					outputs_flip = shuffleLR(flip(outputs_flip[opt.nStack - 1]))
					outputs_flip = Variable(outputs_flip.float())
					if use_gpu:
						outputs_flip = outputs_flip.cuda()
					outputs = (outputs_flip + outputs[opt.nStack - 1]) / 2

				# compute accuracy
				acc_dict_pck, acc_dict_pcp, ae = accuracy(outputs, labels, std)
				#visualize(inputs, labels, outputs, i_batch)
				# statistics
				for i in xrange(opt.nStack):
					running_loss += loss[i].data[0]
				#running_loss += loss.data[0]
				running_acc_pck += acc_dict_pck[0]
				running_acc_pcp += acc_dict_pcp[0]
				running_ae += ae

				running_wrist += (acc_dict_pcp[2] + acc_dict_pcp[5])/2.0
				running_elbow += (acc_dict_pcp[1] + acc_dict_pcp[4])/2.0

				if i_batch % 20 == 0:
					print('epoch: {:4.0f} i_batch: {:4.0f} loss: {:.5f} PCK: {:.4f} PCP: {:.4f} AE: {:.4f} wrist: {:.4f} elbow: {:.4f}'.format(
						epoch, i_batch, running_loss/(i_batch+1), 
						running_acc_pck/(i_batch+1), 
						running_acc_pcp/(i_batch+1), 
						running_ae/(i_batch+1),

						running_wrist/(i_batch+1),
						running_elbow/(i_batch+1)))
				opt.loss[phase].append(running_loss/(i_batch+1))

			epoch_loss = running_loss / data_sizes[phase] * batch[phase]
			epoch_acc_pck = running_acc_pck / data_sizes[phase] * batch[phase] ## the last batch might be smaller
			epoch_acc_pcp = running_acc_pcp / data_sizes[phase] * batch[phase]
			epoch_ae = running_ae / data_sizes[phase] * batch[phase]

			epoch_wrist = running_wrist / data_sizes[phase] * batch[phase]
			epoch_elbow = running_elbow / data_sizes[phase] * batch[phase]

			opt.PCK[phase].append(epoch_acc_pck)
			opt.PCP[phase].append(epoch_acc_pcp)
			opt.AE[phase].append(epoch_ae)

			print('{} Epoch: {:4.0f} Loss: {:.4f} PCK: {:.4f} PCP: {:.4f} AE: {:.4f} wrist: {:.4f} elbow: {:.4f}'.format(
				phase, epoch, epoch_loss, epoch_acc_pck, epoch_acc_pcp, epoch_ae, epoch_wrist, epoch_elbow))
			print('Time: %s'%datetime.datetime.now())

			# deep copy the model
			if phase == 'val' and epoch_acc_pck > best_acc_pck:
				best_acc_pck = epoch_acc_pck
				pck_pcp, pck_ae = epoch_acc_pcp, epoch_ae
				best_model_pck = copy.deepcopy(model)
				torch.save(best_model_pck.state_dict(), '{}/best_model_pck.pth'.format(opt.expDir))
			if phase == 'val' and epoch_acc_pcp > best_acc_pcp:
				best_acc_pcp = epoch_acc_pcp
				pcp_pck, pcp_ae = epoch_acc_pck, epoch_ae
				best_model_pcp = copy.deepcopy(model)

				best_wrist = epoch_wrist
				best_elbow = epoch_elbow

			if phase == 'val' and epoch_ae < best_ae:
				best_ae = epoch_ae
				ae_pck, ae_pcp = epoch_acc_pck, epoch_acc_pcp
				best_model_ae = copy.deepcopy(model)

		if epoch % opt.snapshot == 0:
			torch.save(model.state_dict(), '{}/model_{}_epoch_{}.pth'.format(opt.expDir, opt.nStack, epoch))

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
		time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))
	print('Best PCK: PCK: {:4f} PCP: {:4f} AE: {:4f}'.format(best_acc_pck, pck_pcp, pck_ae))
	print('Best PCP: PCK: {:4f} PCP: {:4f} AE: {:4f} wrist: {:.4f} elbow: {:.4f}'.format(pcp_pck, best_acc_pcp, pcp_ae, best_wrist, best_elbow))
	print('Best AE: PCK: {:4f} PCP: {:4f} AE: {:4f}'.format(ae_pck, ae_pcp, best_ae))
	torch.save(best_model_pck.state_dict(), '{}/best_model_pck.pth'.format(opt.expDir))
	torch.save(best_model_pcp.state_dict(), '{}/best_model_pcp.pth'.format(opt.expDir))
	torch.save(best_model_ae.state_dict(), '{}/best_model_ae.pth'.format(opt.expDir))
	torch.save(stdd, 'stdd.pth')
	return

def demo(data_loaders, data_sizes, model, criterion, optimizer, use_gpu):
	running_acc = 0.0
	model_dir = os.path.join(opt.expDir, opt.preModel)
	f = torch.load(model_dir)
	model.load_state_dict(f)
	model.train(False)
	for i_batch, data_tmp in enumerate(data_loaders['val']):
		# get the inputs
		data, std = data_tmp
		std = std.float()
		inputs, labels = data['image'].float(), data['label'].float()
		# wrap them in Variable
		if use_gpu:
			inputs, labels = inputs.cuda(), labels.cuda()
		inputs, labels = Variable(inputs), Variable(labels)
		# forward
		outputs = model(inputs)
		# outputs = outputs[opt.nStack - 1]
		
		inputs_flip = flip(inputs)
		inputs_flip = inputs_flip.float()
		if use_gpu:
			inputs_flip = inputs_flip.cuda()
		outputs_flip = model(Variable(inputs_flip))
		outputs_flip = shuffleLR(flip(outputs_flip[opt.nStack - 1]))
		outputs_flip = Variable(outputs_flip.float())
		if use_gpu:
			outputs_flip = outputs_flip.cuda()
		outputs = (outputs_flip + outputs[opt.nStack - 1]) / 2
		
		loss = criterion(outputs, labels)
		# compute accuracy
		acc_dict, _, _ = accuracy(outputs, labels, std)
		running_acc += acc_dict[0]
		# statistics
		print('i_batch:', i_batch, 'loss:', loss.data[0], 'PCK:', acc_dict[0])
		#visualize(inputs, labels, outputs, i_batch)
	print(i_batch+1, ' Average PCK: ', running_acc/(i_batch+1))

def load_curve():
	loss = torch.load('loss.pth')
	PCK = torch.load('PCK.pth')
	PCP = torch.load('PCP.pth')
	AE = torch.load('AE.pth')
	plot_curve(loss, PCK, PCP, AE)

if __name__ == '__main__':
	global opt
	opt = get_args()
	set_opt(opt)
	if not os.path.exists('exp'):
		os.mkdir('exp')
	if not os.path.exists('out'):
		os.mkdir('out')
	if opt.plot:
		load_curve()
	else:
		main()
