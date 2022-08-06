# Yolov4 -- neck

​		开始对Yolov4的neck部分进行代码编写与理解.

## 代码实战

### @file: yolo.py

#### 1. CBL

```python
def CBL(filter_in, filter_out, kernel_size, stride = 1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0    # 其实我认为此处可换为 kernel_size // 2
    return nn.Sequential(OrderedDict[
        ('conv', nn.Conv2d(filter_in, filter_out, kernel_size = kernel_size, padding = pad, stride = stride)),
        ('bn', nn.BatchNorm2d(filter_out)),
        ('relu', nn.LeakyReLU(0.1))
    ])
```

#### 2. CBL+Upsample

```python
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        
        self.Upsample = nn.Sequential(
        	CBL(in_channels, out_channels, 1),
            nn.Upsample(scale_factor = 2, mode = 'nearest')
        )
        
    def forward(self, x):
        x = self.Upsample(x)
        return x
```

#### 3. CBL*3

​		若in_filter = 1024, filter的变化为：[`1024 (in_filter)`, 512, 1024, 512]

```python
def make_three_conv(filters_list, in_filters):
    return nn.Sequential(
    	CBL(in_filters, filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
        CBL(filters_list[1], filters_list[2], 1)
    )
```

#### 4. SPP块

​		SPP块中有三个Maxpooling层，提取自己的特征进行自我融合.

```python
class SpatiaPvramidPooling(nn.Module):
    def __init__(self, pool_sizes = [5, 9, 13]):
        super(SpatiaPvramidPooling, self).__init__()
		self.maxpools = nn.ModuleList([
            nn.MaxPool2d(kernel_size = pool_size, stride = 1, padding = kernel_size // 2) for pool_size in pool_sizes
        ])
        
    def forward(self, x):
    	features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim = 1)
        
        return features
```

#### 5. 五次卷积

​		若 in_filter = 512，则filter的变化为：[`512 (in_filter)`,  256, 512, 256, 512, 256], 则 filters_list = [256, 512, 256, 512, 256]

```python
def make_five_conv(filters_list, in_filter):
    return nn.Sequential(
    	CBL(in_filter, filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
		CBL(filters_list[1], filters_list[2], 1),
        CBL(filters_list[2], filters_list[3], 3),
        CBL(filters_list[3], filters_list[4], 1),
    )
```

#### 6. 获取yolov4的输出

```python
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
    	CBL(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1)
    )
    return m
```

#### 7. 构造整个Yolov4的neck_body
