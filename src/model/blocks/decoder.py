from Layers import Up 
from Attentions import AttentionFilter 
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_channel,out_channel,dropout:bool,attention: str,dim_reduction: int):
        super(Decoder,self).__init__()
        """
        Parameters : 
        in_channel -> the number channel input 
        out_channel -> the out number channel 
        attntion -> the type attention mechanism used 
                   attentionFilter : this based on Fast Fourire Transformstin 
                   attentionGate   : this based on Concolution multiplication Kernel 
       dim_reduction : this is tuned paramaters setup based on whihc method attention used in GeneralSegementer model 
       for AttentionGate  dim_reduction = 2 
           AtentionFilter dim_reduction = ( 16 , 4 , 1 )
        
        """
        self.up = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2,padding=0)
        self.reduce = dropout
        self.attention = attention
        self.Att_filter= AttentionFilter(F_g= out_channel, F_l = out_channel, F_int = out_channel // dim_reduction, dim = out_channel // dim_reduction)
        self.Att_gate= Attention_gate(F_g= out_channel, F_l = out_channel, F_int = out_channel // 2)
        self.dropout = nn.Dropout2d(0.5)
        self.Conv = Double_Conv(out_channel+out_channel,out_channel)   
    def forward(self,input_,skip):
        x = self.up(input_)
        if self.attention == "attentionFilter":
            x = self.Att_filter(x,skip) 
        elif self.attention == "attentionGate":
            x = self.Att_gate(x,skip)
        else: 
            print("attention layer not used in Decoder Block ")
        x = torch.cat([x,skip],axis=1) 
        if self.reduce==True:
            x = self.Conv(x)
            x = self.dropout(x)
            return x 
        else:
            x = self.Conv(x)
        return x


