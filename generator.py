import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv3d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

	
class Key_Network(nn.Module):
    def __init__(self, num_filter):
        super(Key_Network, self).__init__()
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        
        # Not finished, check dimemsions
        self.up_1 = upsample_block_2_3d(1, num_filter, act_fn)
        self.up_2 = upsample_block_3d(num_filter, int(num_filter/2), act_fn)
        self.up_3 = upsample_block_3d(int(num_filter/2), int(num_filter/4), act_fn)
        self.up_4 = upsample_block_3d(int(num_filter/4), int(num_filter/8), act_fn)
         
    def forward(self, x):
        up_1 = self.up_1(x)
        up_2 = self.up_2(up_1)
        up_3 = self.up_3(up_2)
        up_4 = self.up_4(up_3)
        return up_1, up_2, up_3, up_4

class Generator(nn.Module):
    def __init__(self, in_dim=1, out_dim=6, num_filter=16):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(self.num_filter, self.num_filter*2, act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(self.num_filter*2, self.num_filter*4, act_fn)
        self.pool_3 = maxpool_3d()
        
        self.bridge = conv_block_2_3d(self.num_filter*4, self.num_filter*8, act_fn)
        
        self.key_generator = Key_Network(self.num_filter*8)
        
        self.trans_1 = upsample_block_3d(self.num_filter*16, self.num_filter*8, act_fn)
        self.up_1 = conv_block_2_3d(self.num_filter*12, self.num_filter*4, act_fn)
        
        self.trans_2 = upsample_block_3d(self.num_filter*8, self.num_filter*4, act_fn)
        self.up_2 = conv_block_2_3d(self.num_filter*6, self.num_filter*2, act_fn)
        
        self.trans_3 = upsample_block_3d(self.num_filter*4, self.num_filter*2, act_fn)
        self.up_3 = conv_block_2_3d(self.num_filter*3, self.num_filter*1, act_fn)
        
        self._output = conv_block_3d(self.num_filter*2, self.num_filter*1, act_fn)
        self.output = conv1_block_3d(self.num_filter*1, self.out_dim, nn.Tanh())
        self.output[0].weight.data.fill_(0.)
        self.output[0].bias.data.fill_(0.)
        print('========================================================')
        print("- Created Generator")

    def forward(self,x,k):
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        
        bridge = self.bridge(pool_3)
        
        key = self.key_generator(k)
        
        morphed_feature_1 = torch.cat([bridge, key[0]], dim=1)
        
        trans_1  = self.trans_1(morphed_feature_1)
        concat_1 = torch.cat([trans_1,down_3], dim=1)
        up_1     = self.up_1(concat_1)
        
        morphed_feature_2 = torch.cat([up_1, key[1]], dim=1)
        
        trans_2  = self.trans_2(morphed_feature_2)
        concat_2 = torch.cat([trans_2,down_2], dim=1)
        up_2     = self.up_2(concat_2)
        
        morphed_feature_3 = torch.cat([up_2, key[2]], dim=1)
        
        trans_3  = self.trans_3(morphed_feature_3)
        concat_3 = torch.cat([trans_3,down_1], dim=1)
        up_3     = self.up_3(concat_3)
        
        morphed_feature_4 = torch.cat([up_3, key[3]], dim=1)
        
        _out = self._output(morphed_feature_4)#.permute(0,2,3,4,1)
        out = self.output(_out)
        out = out.permute(0,2,3,4,1)

        return out[:,:,:,:,0:3], out[:,:,:,:,3:6]

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


def conv_trans_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def upsample_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True),
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def upsample_block_2_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Upsample(scale_factor=(3,2,2), mode='trilinear', align_corners=True),
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model

def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
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
