import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class GlobalFilter(nn.Module):
    def __init__(self, in_channel: int,out_channle: int,dim: int):
        super().__init__()
        self.W = ( out_channle // 2 ) + 1
        self.H = in_channel 
        self.complex_weight = nn.Parameter(torch.randn(self.H, self.W , dim, 2, dtype=torch.float32) * 0.02)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        Frequnecy_dom = x * weight
        x = torch.fft.irfft2(Frequnecy_dom, s=(H, W), dim=(1, 2), norm='ortho')
        return x , Frequnecy_dom

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel,num_groups=2):
        super(Double_Conv,self).__init__()
        
        self.conv = nn.Sequential(
             nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
             nn.GroupNorm(num_groups=num_groups,num_channels=out_channel),
             nn.ReLU(inplace=True),
             nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1),
             nn.GroupNorm(num_groups=num_groups,num_channels=out_channel),
             nn.ReLU(inplace=True))
    def forward(self,input_):
        input_ = self.conv(input_)
        return input_


class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up, self).__init__()
        self.up = nn.Sequential(
               nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2,padding=0)
)
    def forward(self, x):
        x = self.up(x)
        return x

class SoftmaxComplex(nn.Module):
    def __init__(self):
        super(SoftmaxComplex,self).__init__()
        self.Softmax = nn.Softmax2d()
        
    def forward(self,x):
        if x.dtype == torch.complex64:
            B , C , Freq , time_d = x.size()
            activation = self.Softmax(torch.abs(x.view(B*C ,Freq,time_d)))
            return activation.view(B,C , Freq , time_d)  
     
        return self.Softmax(x)

class Out(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Out,self).__init__()
        self.out = nn.Conv2d(in_channel,out_channel,kernel_size=1)
    def forward(self,input_):
        return self.out(input_)
