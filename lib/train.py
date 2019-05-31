import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from lib.eeg_transformer import *

# This masking, combined with fact that the output embeddings are offset by one posi-tion, 
# ensures that the predictions for position i can depend only on the known outputsat positions less than i. 
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# Object for holding a batch of data with mask during training.
class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = torch.ones(src.size(0), 1, src.size(1))
        if trg is not None:
            self.trg = trg[:, :-1,:]
            self.trg_y = trg[:, 1:,:] # used as 'teacher forcing' in training process
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = self.trg_y.size(1)    
    @staticmethod
    # Create a mask to hide padding and future words
    def make_std_mask(tgt, pad): 
        tgt_mask = torch.ones(tgt.size(0), 1, tgt.size(1),dtype = torch.long)
        tgt_mask = tgt_mask.type_as(tgt_mask.data) & subsequent_mask(tgt.size(1)).type_as(tgt_mask.data)
        return tgt_mask

# combine src and tgt as Batch Class
# force the first time-step to be start_symbol which indicates the start of a sequence
def data_gen(dataloader,start_symbol = 1):
    for idx, (data_x,data_y) in enumerate(dataloader):
        data_x[:,0,:] = start_symbol
        src_ = data_x.float()
        data_y[:,0, :] = start_symbol
        tgt_ = data_y.float()
        yield Batch(src_, tgt_, 0)

# run the model and record loss
def run_epoch(data_iter, model,loss_compute):
    total_loss = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y,batch.ntokens)
        total_loss += loss
    return total_loss / (i+1)

def output_prediction(model,ts_test,ls_test, max_len, start_symbol,output_d):
    # we pre-process the data like in data generation
    input_data = torch.from_numpy(ts_test).unsqueeze(0).float()
    input_data[:,0, :] = 1
    true_output = torch.from_numpy(ls_test).unsqueeze(0).float()
    true_output[:,0, :] = 1
    src = input_data.detach()
    tgt = true_output.detach()
    
    test_batch = Batch(src, tgt, 0)
    src = test_batch.src
    src_mask = test_batch.src_mask
    
    model.eval()
    # feed input to encoder to get memory output which is one of the inputs to decoder
    memory = model.encode(src.float(), src_mask)
    ys = torch.ones(1, 1, output_d).fill_(start_symbol)
    
    # apply a loop to generate output sequence one by one. 
    # This means to generate the fourth output we feed the first three generated output 
    # along with memory output from encoder to decoder. 
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        out = model.generator(out)
        # concatenate new output to the previous output sequence
        ys = torch.cat([ys, out[:,[-1],:]], dim=1)
    outs = [out for out in ys]
    # return predicted sequence and true output sequence
    return torch.stack(outs,0).detach().numpy(), true_output