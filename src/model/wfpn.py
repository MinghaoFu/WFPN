import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce
from einops import repeat, reduce, rearrange
import math

def make_model(args):
    return WFPN(args)

def conv(in_feats, out_feats, kernel_size, bias=True):
    return nn.Conv2d(in_feats, out_feats, kernel_size, padding=(kernel_size//2), bias=bias)

def pad_kernel(kernel):
    return F.pad(kernel, (1, 1, 1, 1), 'constant', 0)

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=False):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(shape)
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

def bn_parameters(bn):
    return bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
    
class FMEA(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(FMEA, self).__init__()
        mid_channels = max(in_feats, out_feats) // 16
        if mid_channels == 0:
            mid_channels = 1
        self.sa = nn.Sequential(
            conv(in_feats, 1, 1),
            nn.BatchNorm2d(1),
            conv(1, 1, 3),
            nn.Sigmoid()
        ) 
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv(in_feats, mid_channels, 1),
            nn.ReLU(inplace=True),
            conv(mid_channels, out_feats, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(x)
        return sa, ca

class expander(nn.Module):
    def __init__(self, in_feats, mid_feats):
        super(expander, self).__init__()
        self.conv1 = nn.Conv2d(in_feats, mid_feats, 1)

    def forward(self, x):
        y0 = self.conv1(x)
        return y0

    def param(self):
        return self.conv1.weight, self.conv1.bias

class squeezer(nn.Module):
    def __init__(self, mid_feats, out_feats, conv1_bias):
        super(squeezer, self).__init__()
        self.conv1_bias = conv1_bias
        self.conv3 = nn.Conv2d(mid_feats, out_feats, 3, padding=0)
    
    def forward(self, x):
        # explicitly padding with bias
        y = F.pad(x, (1, 1, 1, 1), 'constant', 0)
        pad = self.conv1_bias.view(1, -1, 1, 1)
        y[:, :, 0:1, :] = pad
        y[:, :, -1:, :] = pad
        y[:, :, :, 0:1] = pad
        y[:, :, :, -1:] = pad

        x = self.conv3(y)
        return x

    def param(self):
        return self.conv3.weight, self.conv3.bias

class projector(nn.Module):
    def __init__(self, hidden_units, factor, proj_type):
        super(projector, self).__init__()
        assert hidden_units // factor > 0 and hidden_units % factor == 0
        self.proj_type = proj_type
        self.out_feats = hidden_units // factor
        self.hidden_unit = hidden_units
        if self.proj_type == 'conv':
            self.conv = conv(hidden_units, self.out_feats, 1)
        elif self.proj_type == 'group conv':
            self.conv = nn.Conv2d(hidden_units, self.out_feats, 1, 1, 1, groups=factor)

    def forward(self, x):
        if self.proj_type == 'avg':
            shape = x.size()
            return x.mean(dim=1).view(shape[0], 1, shape[2], shape[3]).repeat(1, self.out_feats, 1, 1)
        elif self.proj_type == 'max': # non-linear
            shape = x.size()
            max_channel, _ = x.max(dim=1)
            return max_channel.view(shape[0], 1, shape[2], shape[3]).repeat(1,self.out_feats, 1, 1)
        elif 'conv' in self.proj_type:
            return self.conv(x)
        else:
            raise ValueError('Unkown projector type')

    def param(self):
        if self.proj_type == 'conv':
            return pad_kernel(self.conv.weight), self.conv.bias
        elif self.proj_type == 'avg':
            dummy_param = nn.Parameter(torch.empty(0))
            device = dummy_param.device
            return pad_kernel(torch.ones(self.out_feats, self.hidden_unit, 1, 1, device=device)), torch.zeros(self.out_feats, device=device)

class WFP(nn.Module):
    def __init__(self, in_feats, out_feats, factor, proj_type, dropout=0.1):
        super(WFP, self).__init__()
        self.hidden_units = factor * out_feats
        self.expander = expander(in_feats, self.hidden_units)
        self.conv1_bias = self.expander.conv1.bias
        self.bn = nn.BatchNorm2d(self.hidden_units)
        self.projector = projector(self.hidden_units, factor, proj_type)
        self.squeezer = squeezer(self.hidden_units, out_feats, self.conv1_bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b = self.expander(x)
        b = self.bn(b)
        #b = self.dropout(b)
        out = self.squeezer(b) + self.projector(b)
        return out

    def param(self):
        ke, be = self.expander.param()
        rm, rv, eps, bn_w, bn_b = bn_parameters(self.bn)
        ke, be = fuse_conv_bn_weights(ke, be, rm, rv, eps, bn_w, bn_b, False)
        ks, bs = self.squeezer.param()
        kp, bp = self.projector.param()
        device = ke.device

        weight = F.conv2d(input=ks+kp, weight=ke.permute(1, 0, 2, 3))
        _bias = torch.ones(1, self.hidden_units, 3, 3, device=device) * be.view(1, -1, 1, 1)
        bias = F.conv2d(input=_bias, weight=ks+kp).view(-1,) + bs + bp

        return weight, bias

class WFPB(nn.Module):
    def __init__(self, in_feats, out_feats, factor, n_branches, proj_type):
        super(WFPB, self).__init__()
        self.branches = nn.ModuleList()
        for _ in range(n_branches):
            self.branches.append(WFP(in_feats, out_feats, factor, proj_type))
        self.conv3 = conv(in_feats, out_feats, 3)

    def forward(self, x):
        if self.training:
            out = self.conv3(x)
            for branch in self.branches:
                out = out + branch(x) # residual
            return out
        else:
            w, b = self.param()
            out = F.conv2d(input=x, weight=w, bias=b, padding=1)
            return out

    def param(self):
        w, b = self.conv3.weight, self.conv3.bias
        for branch in self.branches:
            _w, _b = branch.param()
            w = w + _w
            b = b + _b
        return w, b

class BLOCK(nn.Module):
    def __init__(self, in_feats, out_feats, factor, n_branches, proj_type):
        super(BLOCK, self).__init__()
        
        self.residual = (in_feats == out_feats)
        self.wfpb = WFPB(in_feats, out_feats, factor, n_branches, proj_type)
        self.fmea = FMEA(in_feats, out_feats)
        self.act = nn.PReLU(out_feats)

    def forward(self, x):
        h = self.wfpb(x)
        sa, ca = self.fmea(x)
        h = h * sa * ca
        if self.residual:
            h += x
        out = self.act(h)
        return out

class WFPN(nn.Module):
    def __init__(self, args):
        super(WFPN, self).__init__()
       
        n_blocks = args.n_blocks
        n_feats = args.n_feats
        n_colors = args.n_colors
        n_branches = args.n_branches
        factor = args.factor
        proj_type = args.proj_type
        self.args = args
        self.scale = args.scale[0]

        backbone = []
        # head
        backbone.append(BLOCK(n_colors, n_feats, factor, n_branches, proj_type))
        # body
        for _ in range(n_blocks):
            backbone.append(BLOCK(n_feats, n_feats, factor, n_branches, proj_type))
        # tail
        backbone.append(nn.Sequential(
            BLOCK(n_feats, n_colors * self.scale ** 2, factor, n_branches, proj_type),
            nn.PixelShuffle(self.scale)
        ))

        self.backbone = nn.Sequential(*backbone)
        
    def forward(self, x):
        #ILR = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.backbone(x)
        return x + ILR

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == "__main__":
    input = torch.rand(1, 3, 5, 5)
    net = WFPB(3, 3, 2, 2, 'conv')
    w, b = net.param()
    original_output = net(input)
    rep_output = F.conv2d(input, w, b, padding=1)
    torch.set_printoptions(precision=20)
    print(original_output[-1][-1][-1][-1])
    print(rep_output[-1][-1][-1][-1])
