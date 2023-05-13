from layers import DoubleConv 
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self,in_channel,out_channel, dropout: bool):
        super(Encoder,self).__init__()
        self.reduce = dropout
        self.dropout = nn.Dropout2d(0.5)
        self.down = nn.Sequential(
            nn.MaxPool2d((2,2)),
            DoubleConv(in_channel,out_channel))         
    def forward(self,input_):
        if self.reduce == True:
            x = self.down(input_)
            x = self.dropout(x)
            return x
        else: 
            x = self.down(input_)
        return x
