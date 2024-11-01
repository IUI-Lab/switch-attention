import os

import torch

from torch.utils.data import DataLoader

import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import utils

from options import parser

from trainer import Trainer
from losses import FocalLoss
from dataset import ASDDatasetSegments
from model import MultipartyTransformer
from imbalanced import ImbalancedDatasetSampler

# reproducibility
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

args = parser.parse_args()
test_sessions = str(args.test_sessions)
behavior_dims = int(args.behavior_dims)
context_segments = int(args.context_segments)
batch_size = int(args.batch_size)
model_name = args.model_name
loss_type = args.loss
epochs = int(args.epochs)
lr = float(args.lr)
split_seed = int(args.split_seed)
oversampling = args.oversampling
numpy_data = args.numpy_data
binary = args.binary

input_dims = 1024 + 768

session_ids = utils.session_ids

for k in range(10):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	criterion = FocalLoss(gamma=10)
	label_levels = 2

	model = MultipartyTransformer(behavior_dims=behavior_dims, input_dims=input_dims, label_levels=label_levels)
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=30)

	print('\nLoading data...')

	train_dataset = []
	val_dataset = []
	test_dataset = []

	for session_id in tqdm(session_ids):
		tmp = ASDDatasetSegments(session_id=session_id, context_segments=context_segments, numpy_data=numpy_data, binary=binary)

		all_idx = [i for i in range(0, len(tmp))]

		if k < 9:
			test_idx = all_idx[k * int(len(tmp) * 10 / 100):(k + 1) * int(len(tmp) * 10 / 100)]
		else:
			test_idx = all_idx[k * int(len(tmp) * 10 / 100):]

		train_idx = [i for i in all_idx if i not in set(test_idx)]

		train, val = train_test_split(torch.utils.data.Subset(tmp, train_idx), test_size=int(len(train_idx) * 20 / 100), random_state=0)
		test = torch.utils.data.Subset(tmp, test_idx)

		train_dataset.append(train)
		val_dataset.append(val)
		test_dataset.append(test)

	train_set = torch.utils.data.ConcatDataset(train_dataset[0:])
	val_set = torch.utils.data.ConcatDataset(val_dataset[0:])
	test_set = torch.utils.data.ConcatDataset(test_dataset[0:])

	if oversampling:
		print('\nCalculating class weights...')
		sampler = ImbalancedDatasetSampler(train_set)
		train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=0)
	else:
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

	val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

	model_unique_name = "{model}".format(model=model_name + "_k{}_lr{}_loss{}".format(k, lr, loss_type))

	model_dir = './models'
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	log_dir = './logs'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	log_path = log_dir + "/{model}.log".format(model=model_unique_name)
	json_path = log_dir + "/{model}.json".format(model=model_unique_name)
	model_path = model_dir + "/{model}.pth".format(model=model_unique_name)

	print('\nTraining...')
	trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, model_path=model_path, log_path=log_path, json_path=json_path, args=args)
	trainer.fit(train_loader, val_loader, test_loader, epochs)