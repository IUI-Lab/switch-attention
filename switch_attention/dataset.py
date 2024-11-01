import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils

# reproducibility
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

class ASDDatasetSegments(Dataset):
	def __init__(self, session_id, context_segments, numpy_data, binary):
		self.data_path = '../data/segments'
		self.session_id = session_id

		# load from numpy
		if numpy_data:
			self.speaker = np.load(os.path.join(os.path.realpath(self.data_path), 'v2', 'c' + str(context_segments), session_id, session_id + '_ALL-speaker.npy'), allow_pickle=True)

			if binary:
				self.target_labels = np.load(os.path.join(os.path.realpath(self.data_path), 'v2', 'c' + str(context_segments), session_id, session_id + '_T01-2-class-labels.npy'))
			else:
				self.target_labels = np.load(os.path.join(os.path.realpath(self.data_path), 'v2', 'c' + str(context_segments), session_id, session_id + '_T01-3-class-labels.npy'))

			self.target_marlin_features = np.load(os.path.join(os.path.realpath(self.data_path), 'v2', 'c' + str(context_segments), session_id, session_id + '_T01-marlin.npy'))
			self.target_trillsson_features = np.load(os.path.join(os.path.realpath(self.data_path), 'v2', 'c' + str(context_segments), session_id, session_id + '_T01-trillsson.npy'))

			self.participant1_marlin_features = np.load(os.path.join(os.path.realpath(self.data_path), 'v2', 'c' + str(context_segments), session_id, session_id + '_P01-marlin.npy'))
			self.participant1_trillsson_features = np.load(os.path.join(os.path.realpath(self.data_path), 'v2', 'c' + str(context_segments), session_id, session_id + '_P01-trillsson.npy'))

			self.participant2_marlin_features = np.load(os.path.join(os.path.realpath(self.data_path), 'v2', 'c' + str(context_segments), session_id, session_id + '_P02-marlin.npy'))
			self.participant2_trillsson_features = np.load(os.path.join(os.path.realpath(self.data_path), 'v2', 'c' + str(context_segments), session_id, session_id + '_P02-trillsson.npy'))

			self.participants_marlin_features = np.stack((self.participant1_marlin_features, self.participant2_marlin_features), axis=1)
			self.participants_trillsson_features = np.stack((self.participant1_trillsson_features, self.participant2_trillsson_features), axis=1)

			self.duration_segments = len(self.target_labels)

	def __len__(self):
		return self.duration_segments

	def __getitem__(self, idx):
		batch = dict()

		batch['speaker'] = self.speaker[idx]

		batch['target_labels'] = self.target_labels[idx]
		batch['target_marlin'] = self.target_marlin_features[idx, ...]
		batch['target_trillsson'] = self.target_trillsson_features[idx, ...]

		batch['participants_marlin'] = self.participants_marlin_features[idx, ...]
		batch['participants_trillsson'] = self.participants_trillsson_features[idx, ...]

		return batch

if __name__ == '__main__':
	session_id = 'g01-s01'
 
 	# test segments dataset
	dataset = ASDDatasetSegments(session_id=session_id, context_segments=3, numpy_data=False, binary=True)
	data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

	print("SEGMENTS")
	for i, batch in enumerate(data_loader):
		print("\nbatch['id']: {}".format(i))

		print("batch['speaker']: {}".format(batch['speaker']))

		print("batch['target_labels']: {}".format(batch['target_labels'].shape))
		print("batch['target_marlin']: {}".format(batch['target_marlin'].shape))
		print("batch['target_trillsson']: {}".format(batch['target_trillsson'].shape))

		print("batch['participants_marlin']: {}".format(batch['participants_marlin'].shape))
		print("batch['participants_trillsson']: {}\n".format(batch['participants_trillsson'].shape))

		break