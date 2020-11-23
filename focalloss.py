import torch
import torch.nn as nn

class Focal_Loss(nn.Module):
	def __init__(self, alpha, beta, gamma, weights = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]):
		super(Focal_Loss, self).__init__()
		self.weights = weights
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		
	def forward(self, pred, target):
		smooth = 1.
		loss = 0
		num_classes = pred.shape[1]
		
		for i in range(0, num_classes):
			intersection = (pred[:,i,:,:,:] * target[:,i,:,:,:]).sum()
			cross_intersection_1 = ((1-pred[:,i,:,:,:]) * target[:,i,:,:,:]).sum()
			cross_intersection_2 = (pred[:,i,:,:,:] * (1 - target[:,i,:,:,:])).sum()
			TI = (intersection + smooth) / (intersection +  self.alpha * cross_intersection_1 + self.beta * cross_intersection_2 + smooth)
			_loss = torch.pow((1 - TI), 1/self.gamma)
			loss += self.weights[i] * _loss
		return loss
