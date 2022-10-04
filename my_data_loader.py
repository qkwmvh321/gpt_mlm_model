from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

import numpy as np
import copy
import random
import json

import torch

from torch import Tensor
from typing import Tuple
from typing import Optional

class my_data_loader(Dataset):
  
    def __init__(self,batch_size : int, seq_len : int, heads : int):
        
        self.batch_size = batch_size 
        self.seq_len = seq_len 
        self.heads = heads 
        #, preproc=None
        self.data_path = './mlm_data_1001.json'
        
        with open(self.data_path,'r') as f:
            self.data_list = json.load(f)['data']
            
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                        bos_token='</s>', eos_token='<seq>', unk_token='<unk>',
                                        pad_token='<pad>', mask_token='<mask>')
        
        self.pad_id = self.tokenizer('<pad>').input_ids[0]
        self.mask_id = self.tokenizer('<mask>').input_ids[0]
    
    def __len__(self):
        print(len(self.data_list))
        return len(self.data_list)//self.batch_size
    
    def get_pad_mask(self, x :Tensor, seq_len:int):
        
        mask = x== self.pad_id

        remask = torch.repeat_interleave(mask, seq_len, dim=0).view(-1,seq_len, seq_len)
       
        for idx in remask:
            for i,iidx in enumerate(idx[0]):
                if iidx ==torch.tensor(True):
                    idx[i] =True
            
        remask2=torch.repeat_interleave(remask.unsqueeze(1),self.heads,dim=1)

        return remask2 
    
    def swap_mask_token(self, data : list):
        
        maked_data = copy.deepcopy(data)
        per = round((len(data)-2)*0.15)
        position = [random.randrange(2, len(data)-1) for i in range(per)]
        
        for idx in position :
        
            maked_data[idx] = self.mask_id
        
        return maked_data, data
    
    def get_token(self, data: list):
        
        X_data = []
        Y_data = []
        for idx in data:
            
            token_data = self.tokenizer(idx).input_ids
            masked_token ,ori_token = self.swap_mask_token(token_data)
            
            masked_token += [self.pad_id for i in range(self.seq_len-len(masked_token))]
            ori_token += [self.pad_id for i in range(self.seq_len-len(ori_token))]
            
            X_data.append(masked_token)
            Y_data.append(ori_token)
            
        return np.array(X_data), np.array(Y_data)
            
    #get image
    def __getitem__(self,idx : list):
        
        datas = self.data_list[idx*self.batch_size : self.batch_size*(idx+1)]
        input_array, ori_input_array= self.get_token(datas)

        input_tensor = torch.tensor(input_array)
        ori_input_tensor = torch.tensor(ori_input_array)
        
        input_pad_mask = self.get_pad_mask(input_tensor, self.seq_len)
        
        return input_tensor, ori_input_tensor, input_pad_mask
    