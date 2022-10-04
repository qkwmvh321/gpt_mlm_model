from torch import Tensor
from typing import Tuple
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position : int =201):
        super(PositionalEncoding, self).__init__()

        # Not a parameter just 
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        self.dropout = nn.Dropout(0.1)
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        y= x + self.pos_table[:, :x.size(1)].clone()
        y2 = self.dropout(y)
        return y2
    
    
class Embedding_ready(nn.Module):
    
    def __init__(self,max_vocab_len : int, embeding_size : int, seq_len : int):
        super(Embedding_ready,self).__init__()
        
        self.embedding_layer = nn.Embedding(num_embeddings=max_vocab_len,
                                            embedding_dim=embeding_size)
        
        self.position_embedding_layer =PositionalEncoding(embeding_size, seq_len)
        
    def forward(self,x : Tensor):
        
        x_embedding  = self.embedding_layer(x)
        x_pos_embedding = self.position_embedding_layer(x_embedding)

        return x_embedding    
    