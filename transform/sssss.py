import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import time


input_window = 10
output_window = 1
batch_size = 25

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)
df = pd.read_csv("expanded_keff_1000_steps(1).csv")
df.head()

class PositionalEncoding(nn.Module):
	def __init__(self,d_model,max_len = 5000):
		super(PositionalEncoding,self).__init__()
		pe = torch.zeros(max_len,d_model)
		position = torch.arange(0,max_len,dtype = torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
		pe[:,0::2] = torch.sin(position*div_term)
		pe[:,1::2] = torch.cos(position*div_term)
		pe = pe.unsqueeze(0).transpose(0,1)
		self.register_buffer('pe',pe)
	def forward(self,x):
		return x+self.pe[:x.size(0),:]
		
class transformer(nn.Module):
	def __init__(self,feature_size = 250,num_layers=1,dropout=0.1):
		super(transformer,self).__init__()
		self.model_type = "Transformer"
		self.source_mask = None
		self.pos_encoder = PositionalEncoding(feature_size)
		self.encoderLayer = nn.TransformerEncoderLayer(d_model = feature_size,nHead=10,dropout = dropout)
		self.transformer_encoder = nn.TransformerEncoder(self.encoderLayer,num_layers = num_layers)
		self.decoder = nn.Linear(feature_size,1)
		self.init_weights()
	def init_weights(self):
		initrange = 0.1
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange,initrange)
	def forward(self,src):
		if(self.src_mask is None or self.src_mask.size(0)!=len(src)):
			device = src.device
			mask = self._generate_square_subsequent_mask(len(src).to(device))
			self.src_mask =mask
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src,self.src_mask)
		output = self.decoder(output)
		return output
	def _generate_square_subsequent_mask(self,sz):
		mask = (torch.triu(torch.ones(sz,sz))==1).tranpose(0,1)
		mask = mask.float().masked_fill(mask == 0,float("-inf")).masked_fill(mask==1,float(0.0))
		return mask
		
	def create_inout_sequences(input_data,tw):
		input_seq = []
		L = len(input_data)
		for i in range(L-tw):
			train_seq = input_data[i:i+tw]
			train_label = input_data[i+output_window:i+output_window+tw]
			input_seq.append((tran_seq,train_label))
		return torch.FloatTensor(input_seq)
	def get_data(data,split):
		series = data
		split = round(split*len(series))
		train_data = series[:split]
		test_data = series[split:]
		#train_data = train_data.cumsum()
		train_data = 2*train_data
		train_seq = create_inout_sequences(train_data,input_window)
		train_seq = train_seq[:-output_window]
		test_data = create_inout_sequences(test_data,input_window)
		test_data = test_data[:-output_window]
		return train_sequence.to(device),test_seq.to(device)
		
	def get_batch(source,i,batch_size):
		seq_len = min(batch_size,len(source)-1-i)
		data = source[i:i+seq_len]
		inputt = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1))
		targett = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
		return inputt,targett
		
	def train(train_data):
		model.train()
		total_loss = 0.0
		start_time = time.time()
		for batch, i in enumerate(range(0,len(train_data)-1,batch_size)):
			data,targets = get_batch(train_data,i,batch_size)
			optimize.zero_grad()
			output = model(data)
			loss = criterion(output,targets)
			loss.backward()
			torch.nn.utils.clap_grad_norm(model.parameters(),0.7)
			optimizer.step()
			
			total_loss += loss.item()
			log_interval = int(len(train_data)/batch_size/5)
			if(batch%log_interval == 0 and batch>0):
				cur_loss = total_loss / log_interval
				elapsed = time.time() - start_time
				print("| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.10f} | {:5.2f} ms | loss {:5.7f}".format(epoch,batch,len(train_data)//batch_size,scheduler.get_lr()[0],elapsed*1000/log_interval,cur_loss))
				total_loss = 0.0
				start_time = time.time()
				
		
	def evaluate(eval_model,data_source):
		eval_model.eval()
		total_loss = 0.0
		eval_batch_size = 1000
		with torch.no_grad():
			for i in range(0,len(data_source) - 1,eval_batch_size):
				data, targets = get_batch(data_source, i , eval_batch_size)
				output = eval_model(data)
				total_loss += len(data[0])
		
		
