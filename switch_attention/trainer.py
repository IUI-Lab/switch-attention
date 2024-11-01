import torch

import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score

import json

from collections import defaultdict

import utils

from losses import ordinal_regression

# reproducibility
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

class Trainer():
	def __init__(self, model, criterion, optimizer, scheduler, model_path, log_path, json_path, args):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.criterion = criterion.to(self.device)

		self.model_path = model_path
		self.log_path = log_path
		self.json_path = json_path

		self.loss_dict = defaultdict(dict)
		self.loss_dict['loss']['train'] = []
		self.loss_dict['loss']['val'] = []
		self.loss_dict['loss']['test'] = []
		self.loss_dict['macro_f1']['train'] = []
		self.loss_dict['macro_f1']['val'] = []
		self.loss_dict['macro_f1']['test'] = []
		self.loss_dict['weighted_f1']['train'] = []
		self.loss_dict['weighted_f1']['val'] = []
		self.loss_dict['weighted_f1']['test'] = []
		self.loss_dict['acc']['train'] = []
		self.loss_dict['acc']['val'] = []
		self.loss_dict['acc']['test'] = []

		self.train_averagemeter = utils.AverageMeter()
		self.val_averagemeter = utils.AverageMeter()

		self.args = vars(args)
    
	def fit(self, train_loader, val_loader, test_loader, epochs):
		for epoch in range(epochs):
			# train
			train_loss, train_macro_f1, train_weighted_f1, train_acc = self.train(train_loader)

			# validate
			val_loss,  val_macro_f1, val_weighted_f1, val_acc, y_dict = self.validate(val_loader)

			# test
			test_loss, test_macro_f1, test_weighted_f1, test_acc, y_dict = self.validate(test_loader)

			# update loss
			train_loss = round(train_loss, 4)
			val_loss = round(val_loss, 4)
			test_loss = round(test_loss, 4)

			self.loss_dict['loss']['train'].append(train_loss)
			self.loss_dict['loss']['val'].append(val_loss)
			self.loss_dict['loss']['test'].append(test_loss)

			self.loss_dict['macro_f1']['train'].append(train_macro_f1)
			self.loss_dict['macro_f1']['val'].append(val_macro_f1)
			self.loss_dict['macro_f1']['test'].append(test_macro_f1)

			self.loss_dict['weighted_f1']['train'].append(train_weighted_f1)
			self.loss_dict['weighted_f1']['val'].append(val_weighted_f1)
			self.loss_dict['weighted_f1']['test'].append(test_weighted_f1)

			self.loss_dict['acc']['train'].append(train_acc)
			self.loss_dict['acc']['val'].append(val_acc)
			self.loss_dict['acc']['test'].append(test_acc)

			loss_statement = "\tModel at epoch: {}, train loss: {}, val loss: {}, test loss: {}".format(epoch+1, train_loss, val_loss, test_loss)
			macro_f1_statement = "\tModel at epoch: {}, train macro_f1: {}, val macro_f1: {}, test macro_f1: {}".format(epoch+1, train_macro_f1, val_macro_f1, test_macro_f1)
			weighted_f1_statement = "\tModel at epoch: {}, train weighted_f1: {}, val weighted_f1: {}, test weighted_f1: {}".format(epoch+1, train_weighted_f1, val_weighted_f1, test_weighted_f1)
			acc_statement = "\tModel at epoch: {}, train acc: {}, val acc: {}, test acc: {}".format(epoch+1, train_acc, val_acc, test_acc)

			print(loss_statement)
			print(macro_f1_statement)
			print(weighted_f1_statement)
			print(acc_statement)
			print('\n')

			self.curr_val_metric = val_weighted_f1 + val_acc + val_macro_f1

			if epoch == 0:
				self.best_val_metric = self.curr_val_metric

				# save model
				torch.save(self.model.state_dict(), self.model_path)
                
				self.loss_dict['pred'] = torch.concatenate(y_dict['pred']).tolist()
				self.loss_dict['target'] = torch.concatenate(y_dict['target']).tolist()
			else: 
				if self.curr_val_metric > self.best_val_metric:
					self.best_val_metric =  self.curr_val_metric

					# save model
					torch.save(self.model.state_dict(), self.model_path)
                    
					self.loss_dict['pred'] = torch.concatenate(y_dict['pred']).tolist()
					self.loss_dict['target'] = torch.concatenate(y_dict['target']).tolist()
            
			self.scheduler.step()

		with open(self.json_path, 'w') as out_file:
			json.dump(self.loss_dict, out_file)

	def train(self, loader):
		y_dict = {}
		y_dict['target'] = []
		y_dict['pred'] = [] 

		self.model.train()
		self.train_averagemeter.reset()

		for i, batch in enumerate(tqdm(loader)):
			labels = batch['target_labels'].float().to(self.device)
			labels = self._quantize_labels(labels)

			marlin_features = torch.concatenate((batch['target_marlin'].unsqueeze(1), batch['participants_marlin']), dim=1).float().to(self.device)
			trillsson_features = torch.concatenate((batch['target_trillsson'].unsqueeze(1), batch['participants_trillsson']), dim=1).float().to(self.device)
			vad_features = batch['speaker']

			# zero grad
			self.optimizer.zero_grad()

			out = self.model(marlin_features, trillsson_features, vad_features)

			y_dict['pred'].append(out)
			y_dict['target'].append(labels)

			if self.args['loss'] == 'ordinal':
				loss = ordinal_regression(out, labels.long())
                
			if self.args['loss'] != 'ordinal':
				loss = self._compute_loss(out, labels.long())
            
			self.train_averagemeter.update(loss.item())

			# backprop
			loss.backward()

			# params update
			self.optimizer.step()

		preds = torch.concatenate(y_dict['pred'])
		targets = torch.concatenate(y_dict['target'])

		if self.args['loss'] == 'ordinal':
			all_f1, macro_f1, weighted_f1, acc = self._compute_f1_acc(preds, targets, ordinal=False)
		else:
			all_f1, macro_f1, weighted_f1, acc = self._compute_f1_acc(preds, targets, ordinal=False)
         
		return self.train_averagemeter.avg, macro_f1, weighted_f1, acc

	def validate(self, loader):
		y_dict = {}
		y_dict['target'] = []
		y_dict['pred'] = [] 
        
		self.model.eval()
		self.val_averagemeter.reset()

		with torch.no_grad():
			for batch in loader:
				labels = batch['target_labels'].float().to(self.device)
				labels = self._quantize_labels(labels)

				marlin_features = torch.concatenate((batch['target_marlin'].unsqueeze(1), batch['participants_marlin']), dim=1).float().to(self.device)
				trillsson_features = torch.concatenate((batch['target_trillsson'].unsqueeze(1), batch['participants_trillsson']), dim=1).float().to(self.device)
				vad_features = batch['speaker']

				out = self.model(marlin_features, trillsson_features, vad_features)

				y_dict['pred'].append(out)
				y_dict['target'].append(labels)
                
				if self.args['loss'] == 'ordinal':
					loss = ordinal_regression(out, labels.long())
                
				if self.args['loss'] != 'ordinal':
					loss = self._compute_loss(out, labels.long())

				self.val_averagemeter.update(loss.item())

		preds = torch.concatenate(y_dict['pred'])
		targets = torch.concatenate(y_dict['target'])

		if self.args['loss'] == 'ordinal':
			all_f1, macro_f1, weighted_f1, acc = self._compute_f1_acc(preds, targets, ordinal=True, val=False)
		else:
			all_f1, macro_f1, weighted_f1, acc = self._compute_f1_acc(preds, targets, ordinal=False, val=False)
        
		return self.val_averagemeter.avg, macro_f1, weighted_f1, acc, y_dict
    
	def _compute_loss(self, pred, target):
		loss = self.criterion(pred, target)

		return loss

	def _compute_f1_acc(self, pred, target, ordinal=False, val=False):
		pred = pred.detach()
		if ordinal:
			pred = torch.cumprod((pred > 0.5), dim=1).sum(1)
		else: 
			pred = torch.max(pred, dim=pred.dim()-1).indices

		macro_f1 = f1_score(target.cpu(), pred.cpu(), average='macro')
		weighted_f1 = f1_score(target.cpu(), pred.cpu(), average='weighted')
		acc = (pred == target).float().mean()
        
		return f1_score(target.cpu(), pred.cpu(), average=None), round(macro_f1, 4), round(weighted_f1, 4), round(acc.item(), 4)

	def _quantize_labels(self, target):
		target = torch.clip(target, min=0, max=2)
		target = torch.floor(target)

		return target