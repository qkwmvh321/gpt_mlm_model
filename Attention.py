import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor
from typing import Tuple
from typing import Optional


class Multihead_Attention(nn.Module):
    
    def __init__(self,head_size : int, embedding_size : int):
        super(Multihead_Attention,self).__init__()
        
        self.k_size = self.q_size =self.v_size = self.embedding_size = embedding_size
        self.head = head_size
        
        self.q_layer = nn.Linear(embedding_size, embedding_size*self.head)
        self.k_layer = nn.Linear(embedding_size, embedding_size*self.head)
        self.v_layer = nn.Linear(embedding_size, embedding_size*self.head)
        
        self.final_layer = nn.Linear(embedding_size * self.head, self.v_size)
        self.Gelu = nn.GELU()
    def transform_tensor(self, input_tensor : Tensor):
        
        #멀티 헤드를 다시 나눠준다. layer를 통과한값은 정확히는 (head , embeddingsize)기때문에 이것을 다시 head 만큼 나눠준다.
        # 1024*8 = 8184  layer를 통과한 값은 (batch_size , seq_len , 8184(head , embedding size) )
        #이것을 다시 (batch size , head , seq_len , embedding size) 로 변경 시켜준다.

        input_tensor = input_tensor.view(input_tensor.size(0), input_tensor.size(1), self.head, self.q_size)
        
        return input_tensor.transpose(1,2)
    
    def calculate_attention(self, query : Tensor, key_T : Tensor, value : Tensor, mask:Optional[Tensor]=None):
        
        #query와 key를 matmul을 한다 이때 key는 key_T로 변경해줘야 한다. matmul한 값은 루트 embeddingsize로 나눠서 scaling한다.
        atten_score = torch.matmul(query,key_T)/ (self.embedding_size**(1/2))
        
        if mask is not None:
            #tri mask 적용
            tri_mask = torch.tril(atten_score)
            
            #큰 -값을 주면 softmax에서는 0값으로 반영 되어진다. 
            masked_atten_score = atten_score.masked_fill(mask==True, -1e9)
            masked_atten_score = masked_atten_score.masked_fill(tri_mask==0., -1e9)
            
        soft_atten = F.softmax(masked_atten_score,dim=-1)
        
        attention = torch.matmul(soft_atten, value)
        attention = attention.transpose(1,2)
                
        return attention
    
    def forward(self, input_em : Tensor, mask:Optional[Tensor]=None):
        
        query = self.q_layer(input_em)
        key = self.k_layer(input_em)
        value  = self.v_layer(input_em)
        
        query = self.transform_tensor(query)
        key = self.transform_tensor(key)
        value = self.transform_tensor(value)
        key_T = key.transpose(2,3)
        if mask is not None:
            attention = self.calculate_attention(query, key_T, value, mask)
        else:
            attention = self.calculate_attention(query, key_T, value)
        
        attention = attention.contiguous().view(input_em.size(0), -1,self.head*self.embedding_size)
        
        
        final_embedding = self.final_layer(attention)
        final_embedding = self.Gelu(final_embedding)
        
        return final_embedding
    