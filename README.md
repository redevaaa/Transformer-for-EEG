# Transformer for EEG

The code has following structure:

1. Whole Model structure(model+optimizer+loss+data batch&masking)
2. Generating Data set (toy data)
3. Training Data (toy data)
4. Predict Data (toy data)
5. Generating Data set (eeg data)
6. Training Data (eeg data)
7. Predict Data (eeg data)

##  Training

   ***All training part is written in the 'Training For … Data Cell'***

1. I use ***Batch Class*** to store data along with corresponding masks. If you want to use your own dataset, you have to create a similar data_gen function as below to transfer raw input data into Batch data. Start symbol is your own choice.

   ```python
   def eeg_data_gen(dataloader,start_symbol = 1):
       "combine src and tgt as Batch Class"
       for idx, (data_x,data_y) in enumerate(dataloader):
           data_x[:,0,:] = start_symbol
           src_ = Variable(data_x.float(), requires_grad=False)
           data_y[:,0, :] = start_symbol
           tgt_ = Variable(data_y.float(), requires_grad=False)
           yield Batch(src_, tgt_, 0)
   ```

2. You can set the model parameters in the cell like below.

```python
# Model options
opt = {}
opt['Transformer-layers'] = 4
opt['Model-dimensions'] = 256
opt['feedford-size'] = 512
opt['headers'] = 8
opt['dropout'] = 0.1
opt['src_d'] = 10
opt['tgt_d'] = 2
```

3. If you want to use CUDA to train, adjust 2 places. First is in ***Class Batch*** cell, uncomment the part with CUDA. Second is in running part, simply uncommen ***model.cuda()***
   ​

## Predict Data

 ***All Prediction part is written in the 'Prediction For … Data Cell'***

1. first step is the same as above. But in this version you can only set the batch size to 1. Thus the data is [timesteps,dimensions] instead of [batch,timesteps,dimensions]
2. using funcion viz to visualize the predicted daya vs true test data.

## Important

1. Transformer uses 'teacher forcing'. In training, it'd better to give the model true output but mask the future part. In this way, model will not walk on the wrong track for a long time which is meaningless. Thus the evaluation part in training is totally different in prediction part. In training, for example, you input 'abcde' and want to output '12345'. The model first get 'a' and output 'out1', then the model input true '1' instead of 'out1'. But in prediction, the model just input what it outputs.
2. Keep in mind that the input in the training part is ' input data + output data + input mask + output mask' . 
3. CUDA works well on training part. But in prediction, even though I just predict one batch, the memory immediately run out. I still cannot figure the reason why. 

