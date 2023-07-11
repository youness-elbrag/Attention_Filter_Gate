from Layers import GlobalFilter , NormLayer
from non_activation import SigmoidComplex , SoftmaxComplex

import torch.nn as nn 
import numpy as np 

class AttentionFilter(nn.Module):
    def __init__(self, F_g, F_l, F_int,dim):
        super(AttentionFilter,self).__init__()
        
        self.Gate_signle = GlobalFilter(F_l,F_int,dim)        
        self.Wieghts_signle = GlobalFilter(F_g,F_int,dim)
        self.Softmax = SigmoidComplex()
        self.NormLayer = LayerNorm(F_int)
                                    
    def forward(self, g, x):
        B , C , W ,H = x.size()
        x1 , g1_feq = self.Gate_signle(g)
        x2 , x1_feq = self.Wieghts_signle(x)
        norm_frwq = self.Varaince(g1_feq)
        atten = self.Softmax((g1_feq.transpose(2,3) @ x1_feq) / torch.sqrt(2* torch.pi * norm_frwq))
        inverse_atten = torch.fft.irfft2(atten, s=(H, W), dim=(2, 3), norm='ortho')   
        out = self.NormLayer(inverse_atten + x)
        return out
    
    def Varaince(self,x):
        x = x.detach().var(-1 , unbiased = False ,keepdims= True)
        return x 

class AttentionFilterGate(nn.Module):
    def __init__(self, F_g, F_l, F_int,dim):
        super(AttentionFilterGate,self).__init__()
        
        self.Gate_signle = GlobalFilter(F_l,F_int,dim)        
        self.Wieghts_signle = GlobalFilter(F_g,F_int,dim)
        self.norm =  nn.BatchNorm2d(F_int)
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.GELU()
                                        
    def forward(self, g, x):
        g1, g1_feq = self.Gate_signle(g)
        norm_g= self.norm(g1)
        x1,x1_feq = self.Wieghts_signle(x)
        norm_x= self.norm(x1)
        psi = self.relu(norm_g + norm_x)

        psi = self.psi(psi)
        out = x * psi
        return out, psi , x1_feq

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
