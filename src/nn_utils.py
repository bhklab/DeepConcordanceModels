from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import sys
import time
import tqdm
import pandas as pd
from collections import defaultdict
from models import TwinNet
from DataClasses import PairwiseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit, KFold
from sklearn.metrics import r2_score

# our own concordance score
from evaluation_metrics import robust_concordance
from scipy.stats import pearsonr

def val_loop(
	model:nn.Module,
	device:str,
	loader:DataLoader,
	optimizer:torch.optim,
	loss_fn:torch.nn
	) -> float:
	

	model.eval()
	running_loss = 0.  
	with torch.no_grad():
		for batch_idx, (x1 , x2 ,y) in enumerate(loader):
			x1,x2,y = x1.to(device), x2.to(device), y.to(device)
		
			
			
			outputs = model(x1, x2)
			outputs = outputs.squeeze()
			
			loss = loss_fn(outputs,y)
			
			# else: 
				
			# 	output1, output2 = model(x1, x2)
			# 	loss = loss_fn(output1,output2,y)
				
			running_loss += loss.item()*x1.shape[0]


		# if batch_idx %10 ==0:
		# 	print(f"batch {batch_idx} loss:\t{loss.item()}")


	return running_loss




def train_loop(
	model:nn.Module, 
	device:str,
	loader:DataLoader,
	optimizer:torch.optim,
	loss_fn:torch.nn,
	prob_outs:bool = True
	) -> float:

	r"""
		Performs a single pass over the pairwise training dataset.

	"""
	
	model.train()
	running_loss = 0.
	

	for batch_idx, (x1 , x2 ,y) in enumerate(loader):
			
		x1,x2,y = x1.to(device), x2.to(device), y.to(device)
		optimizer.zero_grad()
		
		if prob_outs:
			outputs = model(x1, x2)
			outputs = outputs.squeeze()
			
			loss = loss_fn(outputs,y)
			
		else: 
			output1, output2 = model(x1, x2)
			loss = loss_fn(output1,output2,y)
			
		running_loss += loss.item()*x1.shape[0]
		loss.backward()
		optimizer.step()


	return running_loss










