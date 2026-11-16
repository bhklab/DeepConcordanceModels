import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple
from torch.utils.data import Dataset
import pandas as pd
import abc 
import logging
import numpy as np
import abc
import sys





class TwinNet(nn.Module):

	r""" 
	The baseline dual-convergence neural architecture. 

	We build a neural network. 

 	"""
	def __init__(self,
		in_dim:int, 
		hidden_sizes:List[int]=[],
		out_dim:int = 1,
		bias:bool = True,
		drop_prob:float = 0.,
		use_batch_norm:bool = False):

		super(TwinNet,self).__init__()
		

		# print("\n*\n*\n* Initializing Model\n*\n*\n*")
		self.bias = bias
		
	
		
		layers = nn.ModuleList()

		self.bias = bias
		

		
		layer_sizes = [in_dim] + hidden_sizes

		for i in range(len(layer_sizes)-1):
			layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1],bias = self.bias))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(p = drop_prob))
			if use_batch_norm:
				layers.append(nn.BatchNorm1d(layer_sizes[i+1]))

		layers.append(nn.Linear(layer_sizes[-1],out_dim, bias=self.bias))
		
		self.layers = nn.Sequential(*layers)	
		
		
	
	def single_forward_pass(self,
		X:torch.Tensor,
		verbose:bool= False)->torch.Tensor:
		output = self.layers(X)
		return output 


	def forward(self,
		x1:torch.Tensor,
		x2:torch.Tensor
		)->torch.Tensor:

		o1 = self.single_forward_pass(x1)
		o2 = self.single_forward_pass(x2)
	
		o = torch.sigmoid(o1-o2)
		return o








