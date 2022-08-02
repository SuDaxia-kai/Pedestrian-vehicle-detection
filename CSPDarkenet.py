import torch
import torch.nn.functional as F
import torch.nn as nn


#-------------------------------------------------------#
#   MISH激活函数
#-------------------------------------------------------#
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        
    def forward(self, x):
        # F.softplus(x) = torch.log(1 + torch.exp(x))
        return x*torch.tanh(F.softplus(x))
    
    
#-------------------------------------------------------#
#   CBM卷积块
#   CONV+BATCHNORM+MISH
#-------------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super(BasicConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias = False)
        self.bh = nn.BatchNorm2d(out_channels)
        self.activation = Mish()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bh(x)
        x = self.activation(x)
        
        return x
    
#-----------------------------------------------------------#
#   CSPdarknet的结构块组成部分
#   内部堆叠的残差块
#-----------------------------------------------------------#
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels = None):
        super(Resblock, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )
    
    def forward(self, x):
        return x + self.block(x)
    

#------------------------------------------------------------------------------------#
#   CSPNetX模块
#   存在一个大残差边，这个大残差边绕过了很多的残差结构
#------------------------------------------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
        
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride = 2)
        
        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels = out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
        
            self.blocks_conv = nn.Sequential(
                *[Resblock(channels = out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels, 1)
            )
            
    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x)
        
        x = torch.cat([x1, x0], dim = 1)
        x = self.concat_conv(x)
    




