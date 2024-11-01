import torch
import torchvision
import torch.utils.data

import pandas as pd

# reproducibility
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	source: https://github.com/ufoym/imbalanced-dataset-sampler

	Arguments:
		indices: a list of indices
		num_samples: number of samples to draw
		callback_get_label: a callback-like function which takes two arguments - dataset and index
	"""

	def __init__(
		self,
		dataset,
		labels: list = None,
		indices: list = None,
		num_samples: int = None,
		callback_get_label = None,
	):
		# if indices is not provided, all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) if indices is None else indices
        
		# define custom callback
		self.callback_get_label = callback_get_label

		# if num_samples is not provided, draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices) if num_samples is None else num_samples

		# distribution of classes in the dataset
		df = pd.DataFrame()
		df["label"] = self._get_labels(dataset) if labels is None else labels
		df.index = self.indices
		df = df.sort_index()

		label_to_count = df["label"].value_counts()

		weights = 1.0 / label_to_count[df["label"]]

		self.weights = torch.DoubleTensor(weights.to_list())

	def _get_labels(self, dataset):
		if self.callback_get_label:
			return self.callback_get_label(dataset)
		elif isinstance(dataset, torchvision.datasets.MNIST):
			return dataset.train_labels.tolist()
		elif isinstance(dataset, torchvision.datasets.ImageFolder):
			return [x[1] for x in dataset.imgs]
		elif isinstance(dataset, torchvision.datasets.DatasetFolder):
			return dataset.samples[:][1]
		elif isinstance(dataset, torch.utils.data.Subset):
			return dataset.dataset.imgs[:][1]
		elif isinstance(dataset, torch.utils.data.ConcatDataset):
			return self._get_all_labels(dataset)
		else:
			return self._get_all_labels(dataset)

	def _get_all_labels(self, dataset):
		loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)

		for idx, batch in enumerate(loader): pass

		return self._quantize_labels(batch['target_labels'])

	def _quantize_labels(self, target):
		target = torch.clip(target, min=0, max=2)
		target = torch.floor(target)

		return target

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples