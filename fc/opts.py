import os
import argparse
import scipy.io

def get_args():
	projectDir = os.path.abspath('.')

	parser = argparse.ArgumentParser(description='PyTorch Hourglass')

	parser.add_argument('--expID',       default='default',  help='Experiment ID')
	parser.add_argument('--dataset',     default='chair',      help='Dataset choice: bed | ***')
	parser.add_argument('--dataDir',     default=(projectDir+'/../data'),    help='Data directory')
	parser.add_argument('--expDir',      default=(projectDir+'/exp'),     help='Experiment directory')
	parser.add_argument('--manualSeed',  default=-1,         help='Manually set RNG seed')
	parser.add_argument('--GPU',         default=1,          help='Default preferred GPU, if set to -1: no GPU')
	parser.add_argument('--finalPredictions', default=False, help='Generate a final set of predictions at the end of training (default no)')
	parser.add_argument('--nThreads',    default=4,          help='Number of data loading threads')
	## Model options
	parser.add_argument('--demo',        default=False,      help='Show the demo of a previously trained model')
	parser.add_argument('--test',        action='store_true',help='Show the demo of a previously trained model')
	parser.add_argument('--pretrain',    action='store_true',help='Train with a previously trained model')
	parser.add_argument('--preModel',    default='best_model_pck.pth', help='Provide the name of a previously trained model')
	parser.add_argument('--netType',     default='hg',       help='Options: hg | hg-stacked')
	parser.add_argument('--loadModel',   default='none',     help='Provide full path to a previously trained model')
	parser.add_argument('--continue',    default=False,      help='Pick up where an experiment left off')
	parser.add_argument('--branch',      default='none',     help='Provide a parent expID to branch off')
	parser.add_argument('--task',        default='pose',     help='Network task: pose | pose-int')
	parser.add_argument('--nFeats',      default= 256,       help='Number of features in the hourglass')
	parser.add_argument('--nStack',      default=2,          help='Number of hourglasses to stack')
	parser.add_argument('--nModules',    default=1,          help='Number of residual modules at each location in the hourglass')
	## Snapshot options
	parser.add_argument('--savemap',     action='store_true',help='Save heatmap')
	parser.add_argument('--plot',        action='store_true',help='Plot acc')
	parser.add_argument('--PCK',         default=None,       help='Save PCK acc')
	parser.add_argument('--PCP',         default=None,       help='Save PCP acc')
	parser.add_argument('--AE',          default=None,       help='Save AE acc')
	parser.add_argument('--loss',        default=None,       help='Save loss')
	parser.add_argument('--snapshot',    default=10,         help='How often to take a snapshot of the model (0 = never)')
	parser.add_argument('--saveInput',   default=False,      help='Save input to the network (useful for debugging)')
	parser.add_argument('--saveHeatmaps',default=False,      help='Save output heatmaps')
	parser.add_argument('--logInterval', default=10,         help='how many batches to wait before logging training status')
	## Hyperparameter options
	parser.add_argument('--LR',          default=4e-6,     help='Learning rate')
	parser.add_argument('--LRdecay',     default=0.0,        help='Learning rate decay')
	parser.add_argument('--momentum',    default=0.0,        help='Momentum')
	parser.add_argument('--weightDecay', default=0.0,        help='Weight decay')
	parser.add_argument('--alpha',       default=0.99,       help='Alpha')
	parser.add_argument('--epsilon',     default=1e-8,       help='Epsilon')
	parser.add_argument('--crit',        default='MSE',      help='Criterion type')
	parser.add_argument('--optMethod',   default='rmsprop',  help='Optimization method: rmsprop | sgd | nag | adadelta')
	parser.add_argument('--threshold',   default=.001,       help='Threshold (on validation accuracy growth) to cut off training early')
	## Training options
	parser.add_argument('--nEpochs',     default=100,        help='Total number of epochs to run')
	parser.add_argument('--trainIters',  default=8000,       help='Number of train iterations per epoch')
	parser.add_argument('--trainBatch',  default=16,          help='Mini-batch size')
	parser.add_argument('--validIters',  default=1000,       help='Number of validation iterations per epoch')
	parser.add_argument('--validBatch',  default=1,          help='Mini-batch size for validation')
	parser.add_argument('--nValidImgs',  default=1000,       help='Number of images to use for validation. Only relevant if randomValid is set to true')
	parser.add_argument('--randomValid', default=False,      help='Whether or not to use a fixed validation set of 2958 images (same as Tompson et al. 2015)')
	##  Data options
	parser.add_argument('--inputRes',    default=256,        help='Input image resolution')
	parser.add_argument('--outputRes',   default=64,         help='Output heatmap resolution')
	parser.add_argument('--interRes',    default=320,        help='Rrojection plane resolution')
	parser.add_argument('--InChannels',  default=4096,       help='Input channels of FC')
	parser.add_argument('--OutChannels', default=14,         help='Output channels of FC')
	parser.add_argument('--scale',       default=.25,        help='Degree of scale augmentation')
	parser.add_argument('--rotate',      default=30,         help='Degree of rotation augmentation')
	parser.add_argument('--hmGauss',     default=1,          help='Heatmap gaussian size')
	parser.add_argument('--nKeypoints',  default=10,         help='Number of keypoints for bed or other things')
	parser.add_argument('--PCPthr',      default=1.5,        help='Threshold for PCP or PCK torso')
	parser.add_argument('--nmiss',       default=None,       help='Number of missing nodes')
	parser.add_argument('--drop',        default=0.5,        help='Dropout rate')
	parser.add_argument('--pca',         action='store_true',help='Use pca bases')
	parser.add_argument('--inmean',      action=None,        help='Store inmean')


	args = parser.parse_args()
	num_kp = {'bed': 10, 'chair': 10, 'sofa': 14, 'swivelchair': 13, 'table': 8, 'flic': 11, 'car': 20}
	pcp_thr = {'bed': 1.5, 'chair': 1.5, 'sofa': 1.5, 'swivelchair': 1.5, 'table': 1.5, 'flic': 0.2, 'car': 1.5}
	inter_heatmap = {'bed': 240, 'chair': 320, 'sofa': 240, 'swivelchair': 320, 'table': 240, 'car': 320}##
	out_channel = {'bed': 14, 'chair': 14, 'sofa': 16, 'swivelchair': 16, 'table': 15, 'car': 14}## car pca
	args.nKeypoints = num_kp[args.dataset]
	args.PCPthr = pcp_thr[args.dataset]
	args.interRes = inter_heatmap[args.dataset]
	args.InChannels = args.InChannels * args.nKeypoints
	args.OutChannels = out_channel[args.dataset]

	args.GPU = int(args.GPU)
	args.nEpochs = int(args.nEpochs)
	args.nStack = int(args.nStack)
	args.trainBatch = int(args.trainBatch)
	args.drop = float(args.drop)
	args.LR = float(args.LR)
	args.nmiss = {}
	args.nmiss[0] = args.nmiss[1] = args.nmiss[2] = args.nmiss[3] = 0
	args.nmiss[4] = args.nmiss[5] = args.nmiss[6] = args.nmiss[7] = 0

	if args.pca:
		ddir = os.path.join(args.dataDir, args.dataset)
		tmp = scipy.io.loadmat(ddir+'/inmean.mat')
		args.inmean = tmp['inmean'].reshape(3, args.nKeypoints)

	return args

def set_opt(value):
	global opt
	opt = value

def get_opt():
	return opt
