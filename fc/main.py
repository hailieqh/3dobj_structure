from __future__ import print_function, division
import os
import shutil
import time
import datetime
import scipy.io
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
	model = ThreeDNet(opt.InChannels, opt.OutChannels)
	if opt.pretrain:
		model_dir = 'best_model_fc.pth'
		fcnet = torch.load(model_dir)
		model.load_state_dict(fcnet)
	criterion = nn.L1Loss()
	######
	optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.LR,
				alpha=opt.alpha, eps=opt.epsilon,
				weight_decay=opt.weightDecay, momentum=opt.momentum)
	#lr_scheduler = torch.optim.ReduceLROnPlateau(optimizer, 'min')
	# use cuda or not
	use_gpu = (opt.GPU != -1) and torch.cuda.is_available()
	if use_gpu == False:
		print('No GPU......')
	else:
		torch.cuda.set_device(opt.GPU)
		model = model.cuda()
		criterion = criterion.cuda()
	# load data
	#data_transform = ToTensor()
	syn_datasets = {x: SYNDataset(opt.dataDir, #transform=data_transform[x],
						datacat=opt.dataset, phase=x) for x in ['train', 'val']}
	batch = {'train': opt.trainBatch, 'val': opt.validBatch}
	data_loaders = {x: DataLoader(syn_datasets[x], batch_size=batch[x], shuffle=True, 
						drop_last=True, num_workers=4) for x in ['train', 'val']}
	data_sizes = {x: (len(syn_datasets[x]) - len(syn_datasets[x]) % batch[x]) for x in ['train', 'val']}

	# testing
	if opt.test:
		demo(data_loaders, data_sizes, batch, model, criterion, optimizer, use_gpu)
		return

	# training
	opt.acc = {x: [] for x in ['train', 'val']}
	opt.loss = {x: [] for x in ['train', 'val']}
	train_val(data_loaders, data_sizes, batch, model, criterion, optimizer, use_gpu)
	torch.save(opt.acc, 'acc.pth')
	torch.save(opt.loss, 'loss.pth')

def train_val(data_loaders, data_sizes, batch, model, criterion, optimizer, use_gpu):
	since = time.time()

	best_model = model
	best_loss = 10000.0
	#best_acc = 0.0

	for epoch in xrange(opt.nEpochs):
		print('Epoch {}/{}'.format(epoch, opt.nEpochs - 1))
		print('-' * 30)
		for phase in ['train', 'val']:
			if phase == 'train':
				#lr_scheduler.step()
				model.train(True) # set model to training mode
			else:
				model.train(False) # set model to evaluation mode
				result = np.zeros((data_sizes[phase], opt.OutChannels))
				gt = np.zeros((data_sizes[phase], opt.OutChannels))
				result_post = np.zeros((data_sizes[phase], opt.OutChannels))

			running_loss = 0.0
			#running_acc = 0.0
			# iterate over data
			for i_batch, data in enumerate(data_loaders[phase]):
				# get the inputs
				inputs, labels = data['image'].float(), data['label'].float()

				# wrap them in Variable
				if use_gpu:
					inputs, labels = inputs.cuda(), labels.cuda()
				inputs, labels = Variable(inputs), Variable(labels)

				# forward
				outputs = model(inputs)
				loss = criterion(outputs, labels)

				# backward & optimize, only if in training phase
				if phase == 'train':
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				else:
					#lr_scheduler.step(loss)
					out_unnorm = projection(outputs, opt.dataset, opt.interRes, opt.interRes,
											opt.outputRes, opt.outputRes, False, opt.hmGauss)
					out_post = postprocess(out_unnorm, opt.dataset)
					label_unnorm = projection(labels, opt.dataset, opt.interRes, opt.interRes,
											opt.outputRes, opt.outputRes, False, opt.hmGauss)
					result[i_batch, :] = out_unnorm.data[0].cpu().numpy()
					gt[i_batch, :] = label_unnorm.data[0].cpu().numpy()
					result_post[i_batch, :] = out_post.data[0].cpu().numpy()


				#acc = recall(outputs, labels)

				running_loss += loss.data[0]
				#running_acc += acc

				if i_batch % 30 == 0:
					print('epoch: {:4f} i_batch: {:4f} loss: {:.6f}'.format(
						epoch, i_batch, running_loss/(i_batch+1)))
				opt.loss[phase].append(running_loss/(i_batch+1))

			epoch_loss = running_loss / data_sizes[phase] * batch[phase]
			#epoch_acc = running_acc / data_sizes[phase] * batch[phase]
			#opt.acc[phase].append(epoch_acc)
			print('{} Epoch: {:4f} Loss: {:.6f}'.format(
				phase, epoch, epoch_loss))
			print('Time: %s'%datetime.datetime.now())

			# deep copy the model
			if phase == 'val' and epoch_loss < best_loss:
				#best_acc = epoch_acc
				best_loss = epoch_loss
				best_model = copy.deepcopy(model)
				torch.save(best_model.state_dict(), '{}/best_model.pth'.format(opt.expDir))
				scipy.io.savemat('result.mat', {'outputs': result})
				scipy.io.savemat('gt.mat', {'outputs': gt})
				scipy.io.savemat('result_post.mat', {'outputs': result_post})

		if epoch % opt.snapshot == 0:
			torch.save(model.state_dict(), '{}/model_epoch_{}.pth'.format(opt.expDir, epoch))

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
		time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))
	print('Best Val Loss: {:6f}'.format(best_loss))
	torch.save(best_model.state_dict(), '{}/best_model.pth'.format(opt.expDir))
	return

def demo(data_loaders, data_sizes, batch, model, criterion, optimizer, use_gpu):
	if opt.dataset == 'car':
		result_car = np.zeros((1000, opt.nKeypoints*3))
		label_car = np.zeros((1000, opt.nKeypoints*3))
		count_car = 0
	result = np.zeros((data_sizes['val'], opt.OutChannels))
	gt = np.zeros((data_sizes['val'], opt.OutChannels))
	model_dir = os.path.join(opt.expDir, 'best_model.pth')
	f = torch.load(model_dir)
	model.load_state_dict(f)
	model.train(False)
	for i_batch, data in enumerate(data_loaders['val']):
		# get the inputs
		inputs, labels = data['image'].float(), data['label'].float()
		# wrap them in Variable
		if use_gpu:
			inputs, labels = inputs.cuda(), labels.cuda()
		inputs, labels = Variable(inputs), Variable(labels)
		
		# forward
		outputs = model(inputs)
		loss = criterion(outputs, labels)

		#lr_scheduler.step(loss)
		_, out3RT, out_unnorm = projection(outputs, opt.dataset, opt.interRes, opt.interRes,
								opt.outputRes, opt.outputRes, True, opt.hmGauss)
		_, label3RT, label_unnorm = projection(labels, opt.dataset, opt.interRes, opt.interRes,
								opt.outputRes, opt.outputRes, True, opt.hmGauss)
		result[i_batch, :] = out_unnorm.data[0].cpu().numpy()
		gt[i_batch, :] = label_unnorm.data[0].cpu().numpy()

		if opt.dataset == 'car':
			for pid in xrange(opt.nKeypoints):
				for pp in xrange(3):
					result_car[count_car, int(pid + pp*opt.nKeypoints)] = out3RT.data[0, pp, pid]
					label_car[count_car, int(pid + pp*opt.nKeypoints)] = label3RT.data[0, pp, pid]
			count_car += 1

		loss = criterion(outputs, labels)
		print('i_batch:', i_batch, 'loss:', loss.data[0])
	scipy.io.savemat('result.mat', {'outputs': result})
	scipy.io.savemat('gt.mat', {'outputs': gt})
	if opt.dataset == 'car':
		scipy.io.savemat('result_car.mat', {'outputs': result_car})
		scipy.io.savemat('label_car.mat', {'outputs': label_car})

def load_curve():
	loss = torch.load('loss.pth')
	acc = torch.load('acc.pth')
	plot_curve(loss, acc)

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
	elif opt.savemap:
		save_heatmap(opt.dataDir, opt.dataset, 'train')
	else:
		main()
