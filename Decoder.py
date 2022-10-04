from torch import Tensor
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Attention import Multihead_Attention
from SubLayer import Position_Wise_Feed_Forward_Layer
from SubLayer import ResNet_Block

class Decoder_Layer(nn.Module):
    
    def __init__(self, embedding_size : int , heads : int):
        super(Decoder_Layer, self).__init__()
        
        self.embedding_size = embedding_size
        self.heads = heads
        
        self.masked_multi_attention = Multihead_Attention(self.heads, self.embedding_size)
        
        self.ffnn = Position_Wise_Feed_Forward_Layer(self.embedding_size)
        self.res_block = ResNet_Block(self.embedding_size)
        
    def forward(self, input_em: Tensor, masked:Optional[Tensor]=None):
        
        
        masked_muti_atten_out = self.masked_multi_attention(input_em, mask=masked)
        
        output = self.res_block(input_em, masked_muti_atten_out)
        
        ffnn_out = self.ffnn(output)
        
        output3 = self.res_block(output,ffnn_out)
        
        return output3
    
class Decoder(nn.Module):

    def __init__(self, seq_len : int ,embedding_size : int, heads: int, n_layer:int):
        super(Decoder,self).__init__()
        
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.heads = heads
        self.n_layer = n_layer 
                
        self.layer_stack = nn.ModuleList([Decoder_Layer(self.embedding_size, self.heads) for i in range(self.n_layer)])
        
        self.layer_norm = nn.LayerNorm(51201, eps=1e-6)
        
        self.final_layer = nn.Linear(self.embedding_size, 51201)
        self.Gelu = nn.GELU()
        
    def forward(self,input_em : Tensor, masked:Optional[Tensor]=None ):
        
        for decoder_layer in self.layer_stack:
            
            input_em = decoder_layer(input_em, masked)
        
        out = self.final_layer(input_em)
        out = self.layer_norm(out)
        out = self.Gelu(out)
        
        return out