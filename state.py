import torch
import torch.nn as nn
import numpy as np
from model import MobileNet

class States(nn.Module): #optimizer.param_groups[0]['lr']
	def __init__(self, num_r=2, num_c=2, r_lr=0.1, c_lr=0.01):
		super(States, self).__init__()
		self.num_states = num_r + num_c
		self.nets = []
		for i in range(num_r):
			model = MobileNet()
			optimizer = torch.optim.SGD(params=model.parameters(), lr=r_lr)
			model.cuda()
			pretrain = torch.load("./models/pretrain.pth")
			model.load_state_dict(pretrain["state_dict"])
			optimizer.load_state_dict(pretrain["optimizer"])
			self.nets.append({"model":model, "optimizer":optimizer})
		for i in range(num_c):
			model = MobileNet()
			optimizer = torch.optim.Adam(params=model.parameters(), lr=c_lr)
			model.cuda()
			pretrain = torch.load("./models/pretrain.pth")
			model.load_state_dict(pretrain["state_dict"])
			# optimizer.load_state_dict(pretrain["optimizer"])
			self.nets.append({"model":model, "optimizer":optimizer})
		model = nn.Sequential(nn.Linear(self.num_states, 1))
		model.cuda()
		self.eval_net = {"model":model, "optimizer":torch.optim.Adam(params=model.parameters(), lr=0.01)}

	def datax_split(self, data):
		return np.vsplit(data[0:(len(data) // self.num_states * self.num_states)], self.num_states)

	def datay_split(self, data):
		return np.hsplit(data[0:(len(data) // self.num_states * self.num_states)], self.num_states)

	def train(self, datax, datay):
		lossfunc = nn.CrossEntropyLoss()
		lossfunc.cuda()
		data = {"x":self.datax_split(datax), "y":self.datay_split(datay)}
		for index, net in enumerate(self.nets):
			output = net["model"](torch.Tensor(data["x"][index]).cuda())
			loss = lossfunc(output, torch.LongTensor(data["y"][index]).cuda())
			net["optimizer"].zero_grad()
			loss.backward()
			net["optimizer"].step()
		

	def eval(self, datax, datay, epoches=100):
		lossfunc = nn.MSELoss()
		lossfunc.cuda()
		datain = np.zeros((datax.shape[0], self.num_states))
		datay.resize(datay.shape[0], 1)
		for i, net in enumerate(self.nets):
			out = net["model"](torch.Tensor(datax).cuda())
			pred = torch.max(out, 1)[1].cpu().numpy()
			datain[0:datax.shape[0], i] = pred
		for epoch in range(epoches):
			output = self.eval_net["model"](torch.Tensor(datain).cuda())
			loss = lossfunc(output, torch.Tensor(datay).cuda())
			self.eval_net["optimizer"].zero_grad()
			loss.backward()
			self.eval_net["optimizer"].step()
		print(loss.item())

	def save(self):
		for i, net in enumerate(self.nets):
			state = {'model':net["model"].state_dict(), 'optimizer':net["optimizer"].state_dict()}
			torch.save(state, "./models/net." + str(i) + ".pth")
		state = {'model':eval_net["model"].state_dict(), 'optimizer':eval_net["optimizer"].state_dict()}
		torch.save(state, "./models/eval_net.pth")

	def load(self):
		for i, net in enumerate(self.nets):
			checkpoint = torch.load("./models/net." + str(i) + ".pth")
			net["model"].load_state_dict(checkpoint['model'])
			net["optimizer"].load_state_dict(checkpoint['optimizer'])
		checkpoint = torch.load("./models/eval_net.pth")
		eval_net["model"].load_state_dict(checkpoint['model'])
		eval_net["optimizer"].load_state_dict(checkpoint['optimizer'])

	def predict(self, x):
		for net in self.nets:
			out = net["model"](x)
			pred = torch.max(out, 1)[1]
			datain.append(pred)
		datain = np.array(datain)
		return int(self.eval_net["model"](torch.Tensor(datain)))
