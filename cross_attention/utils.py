from collections import defaultdict

# reproducibility
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

openface_features = [' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z', ' gaze_angle_x', ' gaze_angle_y', ' pose_Tx', ' pose_Ty', ' pose_Tz', ' pose_Rx', ' pose_Ry', ' pose_Rz', ' x_1', ' x_2', ' x_3', ' x_4', ' x_5', ' x_6', ' x_7', ' x_8', ' x_9', ' x_10', ' x_11', ' x_12', ' x_13', ' x_14', ' x_15', ' x_16', ' x_17', ' x_18', ' x_19', ' x_20', ' x_21', ' x_22', ' x_23', ' x_24', ' x_25', ' x_26', ' x_27', ' x_28', ' x_29', ' x_30', ' x_31', ' x_32', ' x_33', ' x_34', ' x_35', ' x_36', ' x_37', ' x_38', ' x_39', ' x_40', ' x_41', ' x_42', ' x_43', ' x_44', ' x_45', ' x_46', ' x_47', ' x_48', ' x_49', ' x_50', ' x_51', ' x_52', ' x_53', ' x_54', ' x_55', ' x_56', ' x_57', ' x_58', ' x_59', ' x_60', ' x_61', ' x_62', ' x_63', ' x_64', ' x_65', ' x_66', ' x_67', ' y_0', ' y_1', ' y_2', ' y_3', ' y_4', ' y_5', ' y_6', ' y_7', ' y_8', ' y_9', ' y_10', ' y_11', ' y_12', ' y_13', ' y_14', ' y_15', ' y_16', ' y_17', ' y_18', ' y_19', ' y_20', ' y_21', ' y_22', ' y_23', ' y_24', ' y_25', ' y_26', ' y_27', ' y_28', ' y_29', ' y_30', ' y_31', ' y_32', ' y_33', ' y_34', ' y_35', ' y_36', ' y_37', ' y_38', ' y_39', ' y_40', ' y_41', ' y_42', ' y_43', ' y_44', ' y_45', ' y_46', ' y_47', ' y_48', ' y_49', ' y_50', ' y_51', ' y_52', ' y_53', ' y_54', ' y_55', ' y_56', ' y_57', ' y_58', ' y_59', ' y_60', ' y_61', ' y_62', ' y_63', ' y_64', ' y_65', ' y_66', ' y_67', ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r', ' AU01_c', ' AU02_c', ' AU04_c', ' AU05_c', ' AU06_c', ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c', ' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c', ' AU25_c', ' AU26_c', ' AU28_c', ' AU45_c']

session_ids = ['g01-s01', 'g01-s02', 'g02-s03', 'g02-s04', 'g03-s05', 'g03-s06', 'g04-s07', 'g04-s08', 'g05-s09', 'g05-s10', 'g06-s11', 'g06-s12', 'g07-s13', 'g07-s14', 'g08-s15', 'g08-s16', 'g09-s17', 'g09-s18', 'g10-s19', 'g10-s20', 'g11-s21', 'g11-s22']

support_session_ids = ['g01-s01', 'g02-s03', 'g03-s06', 'g04-s08', 'g05-s09', 'g06-s11', 'g07-s14', 'g08-s16', 'g09-s17', 'g10-s20', 'g11-s21']

public_session_ids = ['g01-s02', 'g02-s04', 'g03-s05', 'g04-s07', 'g05-s10', 'g06-s12', 'g07-s13', 'g08-s15', 'g09-s18', 'g10-s19', 'g11-s22']

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count