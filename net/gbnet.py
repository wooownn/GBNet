import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from net.pvtv2 import pvt_v2_b4

class CAB(nn.Module):

    def __init__(self,lchan,rchan):
        super().__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(lchan, lchan, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(lchan),
            nn.Conv2d(lchan, lchan, kernel_size=1, stride=1,padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(lchan, lchan, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(lchan),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(rchan, rchan, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(rchan),
            nn.Conv2d(rchan, rchan, kernel_size=1, stride=1,padding=0, bias=False),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(rchan, rchan, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(rchan),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                lchan, rchan, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(rchan),
            nn.ReLU(inplace=True),
        )

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        left1 = self.left1(lf)
        left2 = self.left2(lf)

        right1 = self.right1(hf)
        right2 = self.right2(hf)

        left = left1 * torch.sigmoid(right1)
        right = right2 * torch.sigmoid(left2)
        out = self.conv(left + right)
        return out


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    
class BEM(nn.Module):
    def __init__(self):
        super().__init__()

        self.reduce1 = nn.Sequential(
            ConvBNR(64, 32, 3),
            ConvBNR(32, 64, 3),
            ConvBNR(64, 64, 3)
        )
        
        self.reduce2 = nn.Sequential(
            ConvBNR(128, 64, 3),
            ConvBNR(64, 32, 3),
            ConvBNR(32, 64, 3)
        )
        
        self.reduce3 = nn.Sequential(
            ConvBNR(320, 64, 3),
            ConvBNR(64, 32, 3),
            ConvBNR(32, 64, 3)
        )
        
        self.reduce4 = nn.Sequential(
            ConvBNR(512, 64, 3),
            ConvBNR(64, 32, 3),
            ConvBNR(32, 64, 3)
        )
        
        self.x2cb = ConvBNR(64,64,3)
        self.x3cb = ConvBNR(64,64,3)
        self.x4cb = ConvBNR(64,64,3)

        self.conv1x1_1 = ConvBNR(128, 64, 3)
        self.conv1x1_2 = ConvBNR(64, 64, 3)
        self.conv1x = nn.Conv2d(64,64,1,1,0)

        self.conv1y1_1 = ConvBNR(128, 64, 3)
        self.conv1y1_2 = ConvBNR(64, 64, 3)
        self.conv1y = nn.Conv2d(64,64,1,1,0)

        self.conv1z1_1 = ConvBNR(128, 64, 3)
        self.conv1z1_2 = ConvBNR(64, 64, 3)
        self.conv1z = nn.Conv2d(64,64,1,1,0)
        self.block = nn.Sequential(
            ConvBNR(128, 64, 3),
            ConvBNR(64, 32, 3),
            ConvBNR(32, 64, 3),
            ConvBNR(64, 32, 3),
            ConvBNR(32, 64, 3),
            nn.Conv2d(64, 1, 1))

        
    def forward(self, x4, x3, x2, x1):

        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x2 = self.reduce2(x2)
        x3 = self.reduce3(x3)
        x4 = self.reduce4(x4)

        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=False)  # [8, 64, 128, 128]
        x3 = F.interpolate(x3, size, mode='bilinear', align_corners=False)  # [8, 64, 128, 128]
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)  # [8, 64, 128, 128]

        x2 = self.x2cb(x2)
        x3 = self.x3cb(x3)
        x4 = self.x4cb(x4)
    
        x_ori = torch.cat((x1, x2), dim=1)
        x = F.adaptive_avg_pool2d(x_ori, (1, 1)).expand_as(x_ori)
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        x = torch.sigmoid(x)
        x = x2+x1*x
        x = self.conv1x(x)
        # x = x1+x2


        y_ori = torch.cat((x2, x3), dim=1)
        y = F.adaptive_avg_pool2d(y_ori, (1, 1)).expand_as(y_ori)
        y = self.conv1y1_1(y)
        y = self.conv1y1_2(y)
        y = torch.sigmoid(y)
        y = x3+x2*y
        y = self.conv1y(y)
        # y = x2+x3

        z_ori = torch.cat((x3, x4), dim=1)
        z = F.adaptive_avg_pool2d(z_ori, (1, 1)).expand_as(z_ori)
        z = self.conv1y1_1(z)
        z = self.conv1y1_2(z)
        z = torch.sigmoid(z)
        z = x4+x3*z
        z = self.conv1z(z)
        # z = x3+x4
        
        out_1 = torch.cat((x,y),dim=1)
        out_2 = torch.cat((y,z),dim=1)
        out = out_1+out_2
        out = self.block(out)
        return out



class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class BIB(nn.Module):
    def __init__(self, channel):
        super().__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.out_channel = channel//2
        
        self.cv1 = nn.Conv2d(self.channel,self.out_channel,3,1,1)
        self.cv2 = nn.Conv2d(self.channel,self.out_channel,3,1,1)
        self.cv3 = nn.Conv2d(self.channel,self.out_channel,3,1,1)

        self.m = nn.Sequential(*(Bottleneck(self.out_channel, self.out_channel, shortcut=True, e=1.0) for _ in range(1)))
        self.cv4 = nn.Conv2d(self.channel,self.channel,3,1,1)

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c

        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x)
        x3 = self.m(x3)
        
        wei = self.avg_pool(x2)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)

        x1 = x1 * wei 
        x3 = x3 * wei 

        x4 = self.cv4(torch.cat((x1,x3),dim=1))

        return x4


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pvt = pvt_v2_b4()
        path = './models/pvt_v2_b4.pth'
        save_model = torch.load(path)
        model_dict = self.pvt.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.pvt.load_state_dict(model_dict)

        self.bem = BEM()

        self.bib1 = BIB(64)
        self.bib2 = BIB(128)
        self.bib3 = BIB(320)
        self.bib4 = BIB(512)

        self.conv1 = nn.Conv2d(64,1,1,1,0)
        self.conv2 = nn.Conv2d(128,1,1,1,0)
        self.conv3 = nn.Conv2d(320,1,1,1,0)


        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 64)
        self.reduce3 = Conv1x1(320, 64)
        self.reduce4 = Conv1x1(512, 64)

        self.cab1 = CAB(64,64)
        self.cab2 = CAB(64,64)
        self.cab3 = CAB(64,64)


        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(64, 1, 1)
        self.predictor3 = nn.Conv2d(64, 1, 1)


    def forward(self, x): 
        x1, x2, x3, x4 = self.pvt(x)    # [8, 64, 128, 128], [8, 128, 64, 64], [8, 320, 32, 32], [8, 512, 16, 16]
        edge = self.bem(x4, x3, x2, x1) # [8, 1, 128, 128]
        edge_att = torch.sigmoid(edge)  # [8, 1, 128, 128]

        x1a = self.bib1(x1, edge_att)
        x1a_att = torch.sigmoid((self.conv1(x1a)))
        x2a = self.bib2(x2, x1a_att)
        x2a_att = torch.sigmoid((self.conv2(x2a)))
        x3a = self.bib3(x3, x2a_att)
        x3a_att = torch.sigmoid((self.conv3(x3a)))
        x4a = self.bib4(x4, x3a_att)
        
        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)
        x4r = self.reduce4(x4a)
        
        x34 = self.cab3(x3r, x4r)
        x234 = self.cab2(x2r, x3r)
        x1234 = self.cab1(x1r, x2r)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)

        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe
