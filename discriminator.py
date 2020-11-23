import torch
import torch.nn as nn
from densenet import densenet121 as Densenet
from cnn import customized_CNN

class Discriminator(nn.Module):
    def __init__(self, in_dims=1, cnn_arch='customize'):
        super(Discriminator, self).__init__()
        
        if cnn_arch == 'customize':
            self.cnn_backbone = customized_CNN()
        else:
            self.cnn_backbone = Densenet(in_channels=in_dims)
        
        self.l2_norm = l2_normalize_embedding()
            
        self.discriminator = nn.Sequential(nn.Linear(1000,2, bias=True))
        
        self.siamese_discriminator = nn.Sequential(nn.Linear(2000,4000, bias=True),
                                            nn.LeakyReLU(0.2, inplace=True),
                                            nn.Linear(4000,2,bias=True))
        
        print('========================================================')
        print("- Created Discriminator using {}".format(cnn_arch))
    
    def extract_features(self,x):
        _ = self.cnn_backbone(x)
        ft = self.l2_norm(_)
        return ft
    
    def cls(self,x):
        ft = self.extract_features(x)
        cls = self.discriminator(ft)
        return cls
        
    def re_id(self, x_1, x_2):
        ft_1 = self.extract_features(x_1)
        ft_2 = self.extract_features(x_2)
        ft = torch.cat([ft_1,ft_2],dim=1)
        identification = self.siamese_discriminator(ft)
        return identification
                
    def forward(self, x_1, x_2):
        ft_1 = self.extract_features(x_1)
        ft_2 = self.extract_features(x_2)
        ft = torch.cat([ft_1,ft_2],dim=1)
        identification = self.siamese_discriminator(ft)
        cls_1 = self.discriminator(ft_1)
        cls_2 = self.discriminator(ft_2)
        return identification, cls_1, cls_2

        
class Siamese_Discriminator(nn.Module):
    def __init__(self, in_dims=1, cnn_arch='customize'):
        super(Siamese_Discriminator, self).__init__()
        
        if cnn_arch == 'customize':
            self.cnn_backbone = customized_CNN(in_dim=in_dims)
        else:
            self.cnn_backbone = Densenet(in_channels=in_dims)
        
        self.l2_norm = l2_normalize_embedding()
            
        self.siamese_head = nn.Sequential(nn.Linear(2000,4000, bias=True),
                                            nn.LeakyReLU(0.2, inplace=True),
                                            nn.Linear(4000,2,bias=True))
    
    def extract_features(self,x):
        ft = self.l2_norm(self.cnn_backbone(x))
        return ft
        
    def forward(self, x1, x2):
        ft1 = self.extract_features(x1)
        ft2 = self.extract_features(x2)
        ft = torch.cat([ft1,ft2],dim=1)
        pred = self.siamese_head(ft)
        return pred

class l2_normalize_embedding(nn.Module):
    def __init__(self):
        super(l2_normalize_embedding, self).__init__()
        
    def forward(self, x):
        return nn.functional.normalize(x)



