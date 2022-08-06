from turtle import forward
import torch
import torch.nn as nn
from collections import OrderedDict
from CSPDarkenet import darknet53

# CBL的构建
def conv2d(filter_in, filter_out, kernel_size, stride = 1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0    # 其实此处应该是可以用(kernel_size // 2来代替)
    return nn.Sequential(OrderedDict[
        ('conv', nn.Conv2d(filter_in, filter_out, kernel_size = kernel_size, stride = stride, padding = pad)),
        ('bn', nn.BatchNorm2d(filter_out)),
        ('relu', nn.LeakyReLU(0.1))
    ])
    

#--------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#--------------------------------------------------#
class SpatiaPvramidPooling(nn.Module):
    def __init__(self, pool_sizes = [5, 9, 13]):
        super(SpatiaPvramidPooling, self).__init__()
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size = pool_size, stride = 1, padding = pool_size // 2) for pool_size in pool_sizes
        ])
        
    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features+[x], dim = 1)
        
        return features
    
#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


#---------------------------------------------------#
#   三次卷积块
#   [512, 1024]
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

