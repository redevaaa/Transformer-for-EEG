import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from Train import *
from EEG_Transformer import *


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
