import random
import os

import my_data_loader
import model

import torch.utils.data as data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import itertools

from torch import Tensor

from transformers import PreTrainedTokenizerFast

import mlflow
import mlflow.pytorch 

class gpt_runer :
    
    def __init__(self,vocab_size : int = 51201, 
                      batch_size : int = 30, 
                      heads : int = 8,
                      embedding_size : int = 1024,
                      n_layer : int = 4):
        print(batch_size)
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.heads = heads
        self.embedding_size = embedding_size
        self.n_layer = n_layer
        
        self.load_tokenizer()
        
    def load_tokenizer(self):

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                bos_token='</s>', eos_token='<seq>', unk_token='<unk>',
                                                pad_token='<pad>', mask_token='<mask>')
    
    def load_data(self, batch_size, head_size):

        datas = my_data_loader.my_data_loader(batch_size,100,head_size)
        batch_iterator = data.DataLoader(datas, 1, shuffle=True, num_workers=0)
    
        return batch_iterator
    
    def load_model(self, vocab_size : int, embedding_size : int, heads : int, n_layer : int):
        
        gpt_model =model.my_GPT(vocab_size = vocab_size, seq_len = 100, embedding_size = embedding_size, heads=heads, n_layer=n_layer )
        gpt_model =nn.DataParallel(gpt_model).cuda()
        
        return gpt_model
        
    def check_mlm_output(self, output: Tensor, mlm_labels: Tensor):
            
        n = 0
        output = output.argmax(dim=-1).detach()

        mask_pred = output[n][mlm_labels[n]!=3]
        label = mlm_labels[n][mlm_labels[n]!=3]

        mask_pred_out =  self.tokenizer.decode(mask_pred).replace('<pad>','')
        real_label = self.tokenizer.decode(label)
    
        print('real_label : %s'%(real_label))
        print('pred_mask : %s'%(mask_pred_out))
        
        predict_sentence  = 'real_lable :'+real_label +'  pred_mask : '+ mask_pred_out
        
        with open("pred_sentenc.txt", 'w') as f:
            f.write(predict_sentence)
        mlflow.log_artifact('pred_sentenc.txt')
        
    def test_output(self,model, src_data : Tensor):
        
        n = random.randint(0,10)
        
        data = src_data[n] + [self.bos_id] + [self.seq_id]+[self.mask_id for i in range(99)]
        
        out = model(data, mask=None)
        output = out[:,101:,:].argmax(dim=-1).detach()
        predict_sentence = self.tokenizer.decode(output[0])
        
    def model_save(self, model, epoch, optimizer , loss):
        
        save_dir = './model_file/'
        
        stt={'epoch': epoch ,
             'model':model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
                'losses': loss}

        lastest_model = 'gpt_trained_mdoel'+'.pth.tar'

        epoch_dir = str(epoch%20)

        if not os.path.exists(save_dir+epoch_dir):
            os.mkdir(save_dir+epoch_dir)

        torch.save(stt,save_dir+epoch_dir+'/'+lastest_model)
    

    def train(self):
        
        batch_iterator = self.load_data(self.batch_size, self.heads)
        gpt_model = self.load_model(self.vocab_size, self.embedding_size, self.heads, self.n_layer)
        
        criterion_mlm  = nn.CrossEntropyLoss(ignore_index=3)
        
        T_optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=0.00001)
        scheduler= lr_scheduler.ExponentialLR(T_optimizer, gamma= 0.99)
        
        epoch = 1
        
        with mlflow.start_run():
            gpt_model.train()
            while True:

                for step, idx in enumerate(batch_iterator):

                    T_optimizer.zero_grad()
                    try:
                        input_data, output_data, mask  = idx
                        train_data = input_data[0].cuda()
                        mask = mask[0].cuda()
                        true_data = output_data.clone()[0]
                        true_data = true_data.cuda()

                        mlm_labels = output_data.masked_fill(input_data!=6,3).cuda()

                        output = gpt_model(train_data, mask)

                        loss = criterion_mlm(output.view(-1, 51201), mlm_labels[0].view(-1))
                        mlflow.log_metric('mlm_loss', loss.item())
                        loss.backward()
                        T_optimizer.step()
                        
                        with torch.no_grad():
                            self.check_mlm_output(output,mlm_labels[0])
                            
                            print("batch : %s /  step : %s / loss : %s" %  (epoch, step,loss.item()))
     
                    except:
                        print('error')
                        continue
                    
                scheduler.step()
                self.model_save(gpt_model, epoch, T_optimizer , loss.item())
                
        mlflow.end_run
                
if __name__ == "__main__" :
    
    gpt_run = gpt_runer(vocab_size = 51201, batch_size= 40, heads= 12, embedding_size=768, n_layer= 10)
    mlflow.set_experiment('gpt_mlm_test')
    gpt_run.train()
    
    
    
    
    
    
    

    
        
        
        
