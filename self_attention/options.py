import argparse

# reproducibility
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

parser = argparse.ArgumentParser(description='Parameters for Engagemet Estimation')

parser.add_argument('--model_name', required=True)
parser.add_argument('--test_sessions', default='g01-s01_g01-s02')
parser.add_argument('--context_segments', default=5)
parser.add_argument('--behavior_dims', default=128)
parser.add_argument('--epochs', default=50)
parser.add_argument('--lr',  default=0.0005)
parser.add_argument('--batch_size',  default=64)
parser.add_argument('--split_seed', default=42)
parser.add_argument('--loss', choices=('focal', 'classify', 'weighted_classify'), default='focal')
parser.add_argument('--oversampling', action='store_true')
parser.add_argument('--numpy_data', action='store_true')
parser.add_argument('--binary', action='store_true')