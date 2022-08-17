import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


# output:(B, A, n_ch, H, W)  --->   (B, A, H, W, n_ch) ?
def yolo_decode(output, num_classes, anchors, num_anchors, scale_x_y): # scale_x_y缩放因子的作用是什么
    # 判断device
    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()
        
    n_ch = 4 + 1 + num_classes
    A = num_anchors
    B = output.size(0)
    H = output.size(2)
    W = output.size(3)
    
    output = output.view(B, A, n_ch, H, W).permute(0, 1, 3, 4, 2).contiguous()
    
    bx, by = output[..., 0], output[..., 1]
    bw, bh = output[..., 2], output[..., 3]
    
    det_confs = output[..., 4]    # obj
    cls_confs = output[..., 5:]   # cls
    
    bx = torch.sigmoid(bx)
    by = torch.sigmoid(by)
    bw = torch.exp(bw) * scale_x_y - 0.5 * (scale_x_y - 1)
    bh = torch.exp(bh) * scale_x_y - 0.5 * (scale_x_y - 1)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)
    
    grid_x = torch.arange(W, dtype = torch.float).repeat(1, 3, W, 1).to(device)   # 这里应该是repeat(B, A, W, 1),因为是确认B = 1， A = 3，所以才这样写的吧
    grid_y = torch.arange(H, dtype = torch.float).repeat(1, 3, H, 1).permute(0, 1, 3, 2).to(device)
    bx += grid_x
    by += grid_y
    
    for i in range(num_anchors):
        bw[:, i, :, :] *= anchors[i*2]
        bh[:, i, :, :] *= anchors[i*2+1]
    
    # 拓展为和原来一样的维度
    bx = (bx / W).unsqueeze(-1)
    by = (by / H).unsqueeze(-1)
    bw = (bw / W).unsqueeze(-1)
    bh = (bh / H).unsqueeze(-1)
    
    boxes = torch.cat((bx, by, bw, bh), dim=-1).reshape(B, A * H * W, 4) 
    det_confs = det_confs.unsqueeze(-1).reshape(B, A*H*W, 1)
    cls_confs =cls_confs.reshape(B, A*H*W, num_classes)
    outputs = torch.cat([boxes, det_confs, cls_confs], dim=-1)


    return outputs

class YoloLayer(nn.Module):
    '''Yolo layer
    model_out: while inference, is post-processing inside or outside the model
        true: outside
    '''
    def __init__(self, anchor_mask = [], num_classes = 80, anchors = [], num_anchors = 9, stride = 32, scale_x_y = 1):
        super(YoloLayer, self).__init__() 
        # [6,7,8]
        self.anchor_mask = anchor_mask
        # 类别 --> 这里是coco数据集，所以有80类
        self.num_classes = num_classes
        # [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
        self.anchors = anchors
        # 9 
        self.num_anchors = num_anchors
        # 18 // 9 = 2
        self.anchor_step = len(anchors) // num_anchors
        # 32
        self.stride = stride
        # 1
        self.scale_x_y = scale_x_y
        
    def forward(self, output):
        if self.training:
            return output

        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        # [142, 110, 192, 243, 459, 401]/32 把像素值换算成网格单位
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]
        # output:(B, A*n_ch, H, W) ----> (1, 3*(5+80), 19, 19)
        data = yolo_decode(output, self.num_classes, masked_anchors, len(self.anchor_mask),scale_x_y=self.scale_x_y)
        return data
        