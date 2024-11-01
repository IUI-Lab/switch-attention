import torch
import torch.nn as nn

from layers import *

# reproducibility
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

class MultipartyTransformer(nn.Module):
	def __init__(self, behavior_dims, input_dims, label_levels):
		super().__init__()

		self.num_heads = 4
		self.layers = 4

		self.attn_dropout = 0.25
		self.attn_dropout_t1 = 0.25
		self.attn_dropout_p1 = 0.25
		self.attn_dropout_p2 = 0.25
		self.relu_dropout = 0.25
		self.res_dropout = 0.25
		self.out_dropout = 0
		self.embed_dropout = 0.25
		self.attn_mask = True
        
		output_dim = label_levels

		# temporal convolution
		self.proj_marlin = nn.Conv1d(768, 64, kernel_size=1, padding=0)
		self.proj_trillsson = nn.Conv1d(1024, 64, kernel_size=1, padding=0)

		# cross-attention
		self.trans_t1 = self.get_network(self_type='t1')
		self.trans_p1_with_t1 = self.get_network(self_type='p1_t1')
		self.trans_p2_with_t1 = self.get_network(self_type='p2_t1')

		# temoporal encoding
		self.trans_mem = nn.LSTM(3*self.d_t1, 3*self.d_t1, 1, batch_first=True)

		# classification
		self.classify_layer = nn.Linear(3*self.d_t1 , output_dim)

	def get_network(self, self_type='l', layers=-1):
		if self_type in ['t1', 'p1_t1', 'p2_t1']:
			embed_dim, attn_dropout = self.d_t1, self.attn_dropout_t1
		elif self_type in ['p1', 't1_p1', 'p2_p1']:
			embed_dim, attn_dropout = self.d_p1, self.attn_dropout_p1
		elif self_type in ['p2', 't1_p2', 'p1_p2']:
			embed_dim, attn_dropout = self.d_p2, self.attn_dropout_p2

		elif self_type == 't1_mem':
			embed_dim, attn_dropout = 3*self.d_t1, self.attn_dropout
		elif self_type == 'p1_mem':
			embed_dim, attn_dropout = 3*self.d_p1, self.attn_dropout
		elif self_type == 'p2_mem':
			embed_dim, attn_dropout = 3*self.d_p2, self.attn_dropout
		else:
			raise ValueError("Unknown network type")

		return TransformerEncoder(	embed_dim=embed_dim,
									num_heads=self.num_heads,
									layers=max(self.layers, layers),
									attn_dropout=attn_dropout,
									relu_dropout=self.relu_dropout,
									res_dropout=self.res_dropout,
									embed_dropout=self.embed_dropout,
									attn_mask=self.attn_mask)
    
	def forward(self, marlin_features, trillsson_features):
		x_t1_marlin = marlin_features[:, 0, ...].transpose(1, 2)
		x_t1_trillsson = trillsson_features[:, 0, ...].transpose(1, 2)
		x_p1_marlin = marlin_features[:, 1, ...].transpose(1, 2)
		x_p1_trillsson = trillsson_features[:, 1, ...].transpose(1, 2)
		x_p2_marlin = marlin_features[:, 2, ...].transpose(1, 2)
		x_p2_trillsson = trillsson_features[:, 2, ...].transpose(1, 2)

		# project behavioral features
		proj_x_t1_marlin = self.proj_marlin(x_t1_marlin)
		proj_x_t1_trillsson = self.proj_trillsson(x_t1_trillsson)
		proj_x_p1_marlin = self.proj_marlin(x_p1_marlin)
		proj_x_p1_trillsson = self.proj_trillsson(x_p1_trillsson)
		proj_x_p2_marlin = self.proj_marlin(x_p2_marlin)
		proj_x_p2_trillsson = self.proj_trillsson(x_p2_trillsson)

		proj_x_t1 = torch.cat((proj_x_t1_marlin, proj_x_t1_trillsson), dim=1)
		proj_x_p1 = torch.cat((proj_x_p1_marlin, proj_x_p1_trillsson), dim=1)
		proj_x_p2 = torch.cat((proj_x_p2_marlin, proj_x_p2_trillsson), dim=1)

		proj_x_t1 = proj_x_t1.permute(2, 0, 1)
		proj_x_p1 = proj_x_p1.permute(2, 0, 1)
		proj_x_p2 = proj_x_p2.permute(2, 0, 1)

		# cross-attention
		h_t1_self = self.trans_t1(proj_x_t1, proj_x_t1, proj_x_t1)
		h_p1_with_t1s = self.trans_p1_with_t1(proj_x_p1, proj_x_t1, proj_x_t1)
		h_p2_with_t1s = self.trans_p2_with_t1(proj_x_p2, proj_x_t1, proj_x_t1)

		h_t1s = torch.cat([h_t1_self, h_p1_with_t1s, h_p2_with_t1s], dim=2)

		# temoporal encoding
		h_t1s = self.trans_mem(h_t1s)
		if type(h_t1s) == tuple:
			last_hs = h_t1s[0][-1]
			last_h_t1 = last_hs
  
		output = last_h_t1

		# classification
		output = self.classify_layer(output)

		return output