import torch
import torch.nn as nn

class Dice_Loss(nn.Module):
	def __init__(self, weights = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]):
		super(Dice_Loss, self).__init__()
		self.weights = weights
		
	def forward(self, pred, target):
		smooth = 1.
		loss = 0
		num_classes = pred.shape[1]
		
		for i in range(0, num_classes):
			intersection = (pred[:,i,:,:,:] * target[:,i,:,:,:]).sum()
			union = (pred[:,i,:,:,:] + target[:,i,:,:,:]).sum()
			_loss = 1 - (2 * intersection + smooth)/(union + smooth)
			loss += self.weights[i] * _loss
		return loss
