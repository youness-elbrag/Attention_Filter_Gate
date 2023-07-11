import torch
import torch.nn as nn 
import numpy as np 



class PatchEmbeddings(nn.Module):
    def __init__(self , image_size,size_patch, nu_channel, emd_dim, div=None):
        super().__init__()
        """
            Convert the image into patches and then project them into a vector space.
    
        parameters:
        ----------
            image_size: width and height of image and must fixed lengh
            size_patch: value size we want to set of dimension per Patch 
            channel_im: is channel Image Gray == 1 RGB == 3 
        """
        self.image_size = image_size
        self.size_patch = size_patch
        # computer the number Patches
        if div is not None:
            self.number_patch = ((image_size // size_patch) ** 2) // div
        else :
            self.number_patch = (image_size // size_patch) **2 

        self.emd_dim = self.size_patch * self.size_patch
        # Create the Projection 
        self.proj = nn.Conv2d(nu_channel, self.emd_dim,kernel_size=self.size_patch,stride=self.size_patch)
        
    def forward(self,x):
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size,\
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size}*{self.image_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x 

class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
        
    def __init__(self,emd_dim):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(image_size=emd_dim,size_patch=16, nu_channel=1, emd_dim = 16*16)
        #self.cls_token = nn.Parameter(torch.randn(1, 1, emd_dim))
        self.position_embeddings = \
nn.Parameter(torch.randn(1,1 ,self.patch_embeddings.number_patch, emd_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x)
        positional_encoding = x + self.position_embeddings
        x = self.dropout(positional_encoding)
        return x



    
