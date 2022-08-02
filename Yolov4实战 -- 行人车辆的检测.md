[TOC]



# Yolov4实战 -- 行人车辆的检测

Yolov4论文：https://arxiv.org/pdf/2004.10934.pdf

## 理论基础

### 1. 激活函数 -- MISH

​		这是在Yolov4中提出来的激活函数

### 2. BN源码剖析

​		其中BatchNorm2d是在输入的(N, C, H, W)中的C维度上进行的批量归一化.

```python
class BatchNorm2d(_BatchNorm):
#      Shape:
#      - Input: :math:`(N, C, H, W)`
#      - Output: :math:`(N, C, H, W)` (same shape as input)
#    Examples::

#       >>> # With Learnable Parameters
#       >>> m = nn.BatchNorm2d(100)
#       >>> # Without Learnable Parameters
#       >>> m = nn.BatchNorm2d(100, affine=False)
#       >>> input = torch.randn(20, 100, 35, 45)
#       >>> output = m(input)
    def _check_input_dim(self, input):
    if input.dim() != 4:
        raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
```

​		在训练模式下,才进行移动均值跟移动标准差的计算.

## 代码实战

### 1. CSPDarknet.py

​		第一步就是要构建Yolov4网络中的back_bone部分，即两个主要部分，**CBM**与**CSPX**

#### 1.1 CBM -- 激活函数Mish的构建

```python
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(x):
        return x*tanh(F.softplus(x))
```

#### 1.2 CBM 的构建

```python
#-------------------------------------#
# con+bn+mish
#-------------------------------------#
class CBM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1):
        super(BasicConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kenel_size//2, bias = False) # conv
        self.bn = nn.BatchNorm2d(out_channels) # BN
        self.activation = Mish() # M
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        return x
```

#### 1.3 CSPX 的构建

![](/home/sudaxia/图片/2022-08-02 21-42-27 的屏幕截图.png)

​                     																					CSPX 示意图

- 子模块Res_unit的构建

```python
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels = None)
        super(Resblock, self).__init__()

        if hidden_channels == None:
            hidden_channels = channels

        self.block = nn.Sequential(
            CBM(channels, hidden_channels, 1),
            CBM(hidden_channels, channels, 3)
        )
    
    def forwrd(self, x):
        return x + self.block(x)
```

​		要注意CSPX中，第一个CSPX即CSP1跟其他的CSP块是有区别的，要注意区分开来.

```python
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()
    	
    self.downsample_conv = CBM(in_channels, out_channels, 3, stride = 2)
    
    if first:
		self.split_conv0 = CBM(out_channels, out_channels, 1)
        self.split_conv1 = CBM(out_channels, out_channels, 1)
        self.blocks_conv = nn.Sequential(
        	Resblock(channels = out_channels, hidden_channels = channels//2),
            CBM(out_channels, outchannels, 1)
        )
        self.concat_conv = CBM(out_channls * 2, out_channels, 1)   # 因为后面其实跟着一个 1*1 的卷积核，但是好像和CBM有冲突？
    else:
        self.split_conv0 = CBM(out_channels, out_channels//2, 1)
        self.split_conv1 = CBM(out_channels, out_channels//2, 1)
        
        self.block_conv = nn.Sequential(
        	*[Resblock(channels = out_channels//2) for _ in range(num_blocks)],
            CBM(out_channel//2, out_channels, 1)
        )
    
    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        
        x = torch.cat([x1, x0], dim = 1)
        x = self.concat_conv(x)
        
        return x
```



​	
