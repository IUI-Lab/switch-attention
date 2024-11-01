import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import sklearn
import numpy as np

from tqdm import tqdm

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

##########
# Data
##########
session_ids = utils.session_ids

train_dataset = []
test_dataset = []

print('\nLoading data...')
for session_id in tqdm(session_ids):
	tmp = ASDDatasetSegments(session_id=session_id, context_segments=context_segments, numpy_data=numpy_data, binary=binary)
	input_dims = 1024 + 768

	if test_sessions.split('_')[0] == session_id or test_sessions.split('_')[1] == session_id:
		test_dataset.append(tmp)
	else:
		train_dataset.append(tmp)

train_set = torch.utils.data.ConcatDataset(train_dataset[0:-4])
val_set = torch.utils.data.ConcatDataset(train_dataset[-4:])
test_set = torch.utils.data.ConcatDataset(test_dataset[0:])

test_session_ids = test_dataset[0].session_id + '_' + test_dataset[1].session_id

if oversampling:
	print('\nCalculating class weights...')
	sampler = ImbalancedDatasetSampler(train_set)
	train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=0)
else:
	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

##########
# Loss
##########
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if loss_type == 'classify':
	criterion = nn.CrossEntropyLoss()

	if binary:
		label_levels = 2
	else:
		label_levels = 3

if loss_type == 'weighted_classify':
	y = []
	for i in range(len(train_set)):
		sample = train_set[i]
		y.append(sample['target_labels'])

	class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
	class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

	criterion = nn.CrossEntropyLoss(weight=class_weights)

	if binary:
		label_levels = 2
	else:
		label_levels = 3

if loss_type == 'focal':
	criterion = FocalLoss(gamma=10)

	if binary:
		label_levels = 2
	else:
		label_levels = 3

##########
# Model
##########
model = MultipartyTransformer(behavior_dims=behavior_dims, input_dims=input_dims, label_levels=label_levels)
model.to(device)

##########
# Optimizer
##########
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=30)

##########
# Logs
##########
model_unique_name = "{model}".format(model=model_name + "_test{}_lr{}_loss{}".format(test_session_ids, lr, loss_type))

model_dir = './models'
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

log_dir = './logs'
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

log_path = log_dir + "/{model}.log".format(model=model_unique_name)
json_path = log_dir + "/{model}.json".format(model=model_unique_name)
model_path = model_dir + "/{model}.pth".format(model=model_unique_name)

##########
# Train
##########
print('\nTraining...')
trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, model_path=model_path, log_path=log_path, json_path=json_path, args=args)
trainer.fit(train_loader, val_loader, test_loader, epochs)