from blocks.decoder import Decoder
from blocks.encoder import Encoder 
from blocks.Layer import Out


class UnetFre(nn.Module):
    def __init__(self,in_channel,num_filters,num_classes,attention: str):
        super(UnetFre,self).__init__()
        self.n_= num_filters ## 64 ,128,256,512,1024
        filters = [self.n_,self.n_*2,self.n_*4,self.n_*8,self.n_*16]
        
        """Encoder part """
        self.input_conv = Double_Conv(in_channel,filters[0])
        self.Enc_1 = Encoder(filters[0],filters[1],dropout=True)
        self.Enc_2 = Encoder(filters[1],filters[2],dropout=False)
        self.Enc_3 = Encoder(filters[2],filters[3],dropout=True)
        
        """AttentionFilter"""
        self.attention = attention 
        self.Att_filter= AttentionFilter(F_g= filters[0], F_l = filters[0], F_int = filters[2], dim =filters[2] )
        self.Conv = Double_Conv(filters[1],filters[0])  
        self.Up = Up(filters[1],filters[0])


        """ Bottleneck """
        self.dropout = nn.Dropout2d(0.5)
        self.Bottleneck = nn.Sequential(Double_Conv(filters[3],filters[4]),
                                        nn.MaxPool2d((2,2)))    
        
        """Decoder part """
        self.Dec_1 = Decoder(filters[4],filters[3],dropout=True,attention=self.attention,dim_reduction=16)
        self.Dec_2 = Decoder(filters[3],filters[2],dropout=True,attention=self.attention,dim_reduction=4)
        self.Dec_3 = Decoder(filters[2],filters[1],dropout=True,attention=self.attention,dim_reduction=1)
        self.Dec_4 = Decoder(filters[1],filters[0],dropout=False,attention="empty",dim_reduction=1)

        """Output Reconstrution """
        self.output = Out(filters[0], num_classes)
        self.dropout = nn.Dropout2d(0.5)
        
    def forward(self,input_):
        """"Encoder part """
        en_0 = self.input_conv(input_)
        en_1 = self.Enc_1(en_0)
        en_2 = self.Enc_2(en_1)
        en_3 = self.Enc_3(en_2)
        
        """ Bottleneck """
        neck = self.Bottleneck(en_3)
        """Decoder part """    
        de_1  = self.Dec_1(neck,en_3)
        de_2  = self.Dec_2(de_1,en_2)        
        de_3 = self.Dec_3(de_2,en_1)
        if self.attention == "attentionFilter":
            de_4 = self.Up(de_3)
            x1 = self.Att_filter(de_4,en_0)
            d2 = torch.cat((x1, de_4), dim=1)
            d2 = self.Conv(d2)
            out = self.output(d2)
            return out
        de_4 = self.Dec_4(de_3,en_0)
        
        """Output Reconstrution """
        out = self.output(de_4)
        return out
