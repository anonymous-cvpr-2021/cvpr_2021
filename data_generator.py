import os
import torch
import numpy as np
import nibabel as nib
from random import randint, choice
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultils import *

class ppmi(Dataset):
	def __init__(self, data_folder, mode = 'train'):
		super(ppmi, self).__init__()
		self.scale_factor = 1/2
		self.data_folder = data_folder
		
		self.im_list = os.listdir(self.data_folder)

		print('========================================================')
		print('==> Loaded:"{}"'.format(self.data_folder))
		print('==> Number of image {}'.format(len(self.im_list)))

	def __len__(self):
		return int(len(self.im_list))

				
	def __getitem__(self, index):
		k_1, k_2 = torch.rand(1, 3, 6, 5), torch.rand(1, 3, 6, 5)
		
		im = self.im_list[index]
		x = load_nii_to_tensor(os.path.join(self.data_folder, im, 'T1.nii.gz'))
		y = load_segmap_to_tensor(os.path.join(self.data_folder, im, 'segmap.nii.gz'))
		#resize image
		x = resize_tensor(x, scalefactor=1/2)
		y = correct_seg_gt(resize_tensor(y, scalefactor=1/2))
		return x, y, k_1, k_2, im
		
class ppmi_pairs(Dataset):
	def __init__(self, data_folder, ratio = 0.5):
		super(ppmi_pairs_2, self).__init__()
		self.ratio = ratio
		self.scale_factor = 1/2
		self.data_folder = data_folder
		
		self.im_list = os.listdir(self.data_folder)
		self.positives, self.negatives = get_examples(self.im_list)
		print('========================================================')
		print('==> Loaded:"{}" as {} set'.format(self.data_folder, self.mode))
		print('==> No of positives: {}, No of negatives:{}'.format(len(self.positives),len(self.negatives)))

	def __len__(self):
		return int(len(self.positives)/self.ratio)
	
	def __get_pos_item__(self, index):
		s = 1
		im_1, im_2 = self.positives[index]
		
		x_1 = load_nii_to_tensor(os.path.join(self.data_folder, im_1, 'T1.nii.gz'))
		y_1 = load_segmap_to_tensor(os.path.join(self.data_folder, im_1, 'segmap.nii.gz'))
		
		x_2 = load_nii_to_tensor(os.path.join(self.data_folder, im_2, 'T1.nii.gz'))
		y_2 = load_segmap_to_tensor(os.path.join(self.data_folder, im_2, 'segmap.nii.gz'))

		return	x_1, x_2, y_1, y_2, s
	
	def __get_neg_item__(self, index):
		s = 0
		im_1, im_2 = choice(self.negatives)
			
		x_1 = load_nii_to_tensor(os.path.join(self.data_folder, im_1, 'T1.nii.gz'))
		y_1 = load_segmap_to_tensor(os.path.join(self.data_folder, im_1, 'segmap.nii.gz'))
		
		x_2 = load_nii_to_tensor(os.path.join(self.data_folder, im_2, 'T1.nii.gz'))
		y_2 = load_segmap_to_tensor(os.path.join(self.data_folder, im_2, 'segmap.nii.gz'))

		return	x_1, x_2, y_1, y_2, s
				
	def __getitem__(self, index):
		k_1, k_2 = torch.rand(1, 3, 6, 5), torch.rand(1, 3, 6, 5)
		inter_subject = 0
		intra_subject = 1
		
		im = choice(self.im_list)
		x = load_nii_to_tensor(os.path.join(self.data_folder, im, 'T1.nii.gz'))
		y = load_segmap_to_tensor(os.path.join(self.data_folder, im, 'segmap.nii.gz'))
		#resize image
		x = resize_tensor(x, scalefactor=1/2)
		y = correct_seg_gt(resize_tensor(y, scalefactor=1/2))
		
		if index < len(self.positives):
			idx = index
			x_1, x_2, y_1, y_2, s = self.__get_pos_item__(idx)
			
		else:
			idx = index - len(self.positives)
			x_1, x_2, y_1, y_2, s = self.__get_neg_item__(idx)
		
		#resize image
		x_1 = resize_tensor(x_1, scalefactor=1/2)
		x_2 = resize_tensor(x_2, scalefactor=1/2)
		y_1 = correct_seg_gt(resize_tensor(y_1, scalefactor=1/2))
		y_2 = correct_seg_gt(resize_tensor(y_2, scalefactor=1/2))
		return x, y, x_1, x_2, y_1, y_2, s, k_1, k_2, inter_subject, intra_subject

def resize_tensor(tensor, scalefactor=1/2):
	shape = tensor.shape
	tensor = tensor.view(-1,shape[0],shape[1],shape[2],shape[3])
	tensor = F.interpolate(tensor,scale_factor=scalefactor, mode='trilinear',align_corners=True)
	tensor = tensor.view(shape[0],int(shape[1]/2),int(shape[2]/2),int(shape[3]/2))
	return tensor

def correct_seg_gt(y):
	C, X, Y, Z = y.size()
	_y = y.argmax(dim=0)
	y = torch.zeros(y.shape)
	for i in range(C):
		__y = torch.zeros(_y.shape)
		__y[_y==i] = 1
		y[i,:,:,:] = __y
	return y

def gen_crop_point(brain_mask):
	points = []
	msk = nib.load(brain_mask).get_data()
	for x in range(32,144-32):
		for y in range(32,192-32):
			for z in range(32,160-32):
				if msk[x,y,z] == 0:
					pass
				else:
					x_ = x-32
					y_ = y-32
					z_ = z-32
					points += [(x_,y_,z_)]
	return points


def segmap_to_mask(segmap):
	mask = segmap[0,:,:,:]==0
	shape = mask.shape
	mask = mask.view(-1, shape[0], shape[1], shape[2]).float()
	return mask
