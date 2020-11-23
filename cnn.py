import torch
import torch.nn as nn
import torch.nn.functional as F

class customized_CNN(nn.Module):
    def __init__(self, in_dim=1, out_dim=1000,num_filter=32):
        super(customized_CNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        
        self.down_1 = conv_block_3d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_3d(self.num_filter, self.num_filter*2, act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_3d(self.num_filter*2, self.num_filter*4, act_fn)
        self.pool_3 = maxpool_3d()
        self.down_4 = conv_block_3d(self.num_filter*4, self.num_filter*8, act_fn)
        self.pool_4 = maxpool_3d_2()
        self.down_5 = conv_block_3d(self.num_filter*8, self.num_filter*16, act_fn)
        
        self.fc = fully_connected(self.num_filter*16, out_dim, act_fn)#

    def forward(self,x):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)
        down_5 = self.down_5(pool_4)
        feature = self.fc(down_5).view(x.shape[0],-1)
        return feature

def fully_connected(in_dim,out_dim,act_fn):
	model = nn.Sequential(
		nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=0),
		act_fn)
	return model

def conv1_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool

def maxpool_3d_2():
    pool = nn.AdaptiveAvgPool3d((3,3,3))
    return pool

def conv_block_2_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model    


def conv_block_3_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        conv_block_3d(out_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model
