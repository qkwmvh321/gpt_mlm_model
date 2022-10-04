import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from typing import Tuple
from typing import Optional

class ResNet_Block(nn.Module):
    
    def __init__(self,embedding_size : int):
        super(ResNet_Block,self).__init__()
        
        self.norm_layer = nn.LayerNorm(embedding_size)
        
    def forward(self, x : Tensor, y: Tensor):
        
        y = x +y
        out = self.norm_layer(y)
        
        return out

    
class Position_Wise_Feed_Forward_Layer(nn.Module):
    
    def __init__(self,embedding_size : int ):
        super(Position_Wise_Feed_Forward_Layer,self).__init__()
        
        self.embedding_size = embedding_size
    
        self.first_layer = nn.Linear(self.embedding_size, self.embedding_size*2 )
        self.layer_norm1 = nn.LayerNorm(self.embedding_size*2, eps=1e-6)
        
        self.second_layer = nn.Linear(self.embedding_size*2, self.embedding_size)
        self.layer_norm2 = nn.LayerNorm(self.embedding_size, eps=1e-6)
        
        self.Gelu = nn.GELU()
        self.dropout= nn.Dropout(0.1)
        
    def forward(self,input_data : Tensor):
        
        dff = self.first_layer(input_data)
        dff = self.layer_norm1(dff)
        dff = self.Gelu(dff)
        dff = self.dropout(dff)
        dff2 = self.second_layer(dff)
        dff2 = self.layer_norm2(dff2)
        dff3 = self.Gelu(dff2)
        dff3 = self.dropout(dff3)
        
        return dff3