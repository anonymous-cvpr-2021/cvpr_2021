import torch
import torch.nn as nn
import torch.nn.functional as F

class Gradient_Loss(nn.Module):
	def __init__(self):
		super(Gradient_Loss, self).__init__()
	
	def gradient(self, p):
		#gradient step = 1
		#Compute dp/dx
		px = 	p[:,:-1,:-1,:-1,2]
		px_1 =  p[:, 1:,:-1,:-1,2]
		dp_dx = px_1 - px
		#Compute dp/dy
		py = 	p[:,:-1,:-1,:-1,1]
		py_1 = 	p[:,:-1, 1:,:-1,1]
		dp_dy = py_1 - py
		#Compute dp/dz
		pz = 	p[:,:-1,:-1,:-1,0]
		pz_1 = 	p[:,:-1,:-1, 1:,0]
		dp_dz = pz_1 - pz
		return dp_dx, dp_dy, dp_dz
		
	def forward(self, field):
		dp_dx, dp_dy, dp_dz = self.gradient(field)
		magnitude = dp_dx**2 + dp_dy**2 + dp_dz**2
		return magnitude.mean()

