import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from  tensorboardX import SummaryWriter
class Model(nn.Module):
	def __init__(self,args,device):
		super(Model, self).__init__()
		self.device = device
		self.batch_size = 50
		self.z_size = 128
		self.key_size = 256
		self.hidden_size = 512
		self.max_seq = 100
		self.Encoder = Encoder(args)
		self.M_k = M_k(args, self.batch_size, self.key_size).to(device)
		self.M_v = M_v(args, self.batch_size, self.max_seq, self.z_size).to(device)
		self.controller = controller(self.key_size,self.hidden_size)
		self.key_w_out = nn.Linear(self.hidden_size, self.key_size)
		self.g_out = nn.Linear(self.hidden_size, 1)
		self.confidence_gain = nn.Parameter(torch.ones(1))
		self.confidence_bias = nn.Parameter(torch.zeros(1))
		self.y_out = nn.Linear(self.hidden_size, 33)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)
	def forward(self,x_seq):
		hidden = torch.zeros(1, x_seq.shape[0], self.hidden_size).to(self.device)

		k_read = torch.zeros(x_seq.shape[0], 1, self.key_size + 1).to(self.device)
		y_hat = []
		for t in range(x_seq.shape[2]):
			z_t = self.Encoder(x_seq[:,:,t]).unsqueeze(2)

			rnn_out, hidden = self.controller(k_read,hidden)
			# Gates
			g = self.sigmoid(self.g_out(rnn_out))

			# Task output layer
			y_pred_t = self.y_out(rnn_out).squeeze()
			y_hat.append(y_pred_t.squeeze())

			# read from memory
			k_read = self.read_from_memory(z_t, g, t)
		# Key output layers
		key_w = self.relu(self.key_w_out(rnn_out)).squeeze()
		

		return torch.stack(y_hat,dim=2), key_w

	def read_from_memory(self, z_t, g, t):
		# Read key
		w_k = self.softmax((z_t * self.M_v.data[:,:,t,:]).sum(dim=1))
		c_k = self.sigmoid(((z_t * self.M_v.data[:,:,t,:]).sum(dim=1) * self.confidence_gain) + self.confidence_bias)
		k_read = g * (torch.cat([self.M_k.data, c_k.unsqueeze(1)], dim=1) * w_k.unsqueeze(1)).sum(2).unsqueeze(1)
		return k_read

class Encoder(nn.Module):
	# Encoder; can be a CNN or others
	def __init__(self,args):
		super(Encoder, self).__init__()
		pass
	def forward(self, x, device = None):
		pass
		return x.to(device)

class Memory:
	def __init__(self,args):
		# the volume of the memory
		self.M_vol = args.M_vol
		pass
	def clear(self,x):
		pass
	def to(self,device):
		self.data = self.data.to(device)
		return self
	

class M_v(Memory):
	def __init__(self,args,batch_size,max_seq,z_size):
		super(M_v,self).__init__(args)
		self.data = torch.zeros((batch_size, z_size, max_seq, 2))
		

	def write(self,x_seq):
		data = torch.cat([self.data, x_seq.unsqueeze(-1)],dim=-1)
		self.data = data.detach()
	
	def update(self,loss):
		self.data[:,:,:,-1] *= torch.exp(loss).detach()
	

class M_k(Memory):
	def __init__(self,args,batch_size,key_size):
		super(M_k,self).__init__(args)
		self.data = torch.zeros((batch_size, key_size, 2))
		
		
	def write(self,k_write):
		data = torch.cat([self.data, k_write.unsqueeze(2)],dim=2)
		self.data = data.detach()


class controller(nn.Module):
	def __init__(self,k_size,hidden_size):
		super(controller,self).__init__()
		if args.controller_type == 'RNN':
			self.k_size = k_size
			self.hidden_size = hidden_size
			self.RNN = nn.RNN(self.k_size + 1, self.hidden_size, batch_first=True)
	
	def forward(self,k_read,hidden):
		RNN_out, hidden = self.RNN(k_read, hidden)
		return RNN_out,hidden
class args:
	def __init__(self):
		self.controller_type = 'RNN'
		self.M_vol = 10
		self.batch_size = 50
		self.device = torch.device("cuda:0")
		self.lr = 1e-3



def train():
	device = torch.device("cuda:0")
	model = Model(args,device).to(device)
	loss_fn = nn.MSELoss().to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	with SummaryWriter(comment='model')as w:
		w.add_graph(model,(torch.zeros((50,128,100)).to(device)))		
		for i in range(3):
			x_seq = torch.zeros((50,128,100)).to(device)
			y = torch.zeros((50,33,100)).to(device)
			optimizer.zero_grad()
			y_pred, key_w= model(x_seq)
			loss = loss_fn(y_pred,y)
			w.add_scalar('loss',loss,global_step=i)
			# Update model
			loss.backward(retain_graph=True)
			optimizer.step()
			print(loss)
			model.M_v.write(x_seq)
			model.M_k.write(key_w)
			model.M_v.update(loss)

args = args()

train()