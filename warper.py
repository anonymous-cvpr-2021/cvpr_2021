import torch
import torch.nn as nn
import torch.nn.functional as F

class Warper(nn.Module):
    def __init__(self, device):
        super(Warper, self).__init__()
        self.device = device
        
    def forward(self, x, flow_field):
        size = x.size()
        theta = torch.eye(3,4).repeat(size[0],1).view(-1,3,4)
        grid = F.affine_grid(theta, size).to(self.device)
        grid = grid + flow_field
        return nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros')
