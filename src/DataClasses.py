import time
import torch 
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
import pandas as pd
import abc 
import logging
import utils
import numpy as np
import sys
import bisect
import tqdm
from itertools import combinations, permutations







class PairwiseDataset(Dataset):

	r"""
	
	This class implements a pairwise data loader for the training of a Deep Concordance Model based on
	pharmacogenomic (PGx) data.

	The class takes as an input a matrix of molecular observations and a vector of drug responses and produces a list
	of pairwise data points (x1,x2,target) where x1 and x2 are vectors of omics and the target value to be used for 
	computing a loss function.

	As
	things up a list of pairs and their gaps is maintained in sorted order and processed using the bisect module
	to keep things quick.

	Args:

		X: np.array - The n x F matrix of feature vectors. 
		
		y: np.array - The n x 1 vector of response values. Assumed to be positive reals in [0,1].
		
		delta: Optional[float] - The initial threshold to consider pairs well separated. Need |y[1]-y[2]|>delta
			default: 0.0
		
		tau: Optional[float] - The threshold above which to consider a response significant. 
		
		prob_loss: Optional[bool] - Whether or not we should prepare the data for a probabilistic or ranking loss. 


	"""

	
	

	def __init__(
		self,
		X:np.array,
		y:np.array,
		delta:Optional[float]=0.,
		tau:Optional[float]=0.,
		prob_loss:Optional[bool] = True,
		symmetric_pairs:bool = False
		) -> None:

		self.X = torch.tensor(X,dtype = torch.float32)
		self.y = y
		self.prob_loss= prob_loss
		self.symmetric_pairs = symmetric_pairs
		self._build_gap_lists()
		
		


	
		self.update_data(
			delta = delta,
			tau = tau)
		


	def __len__(self):
		return len(self.data)

	def __getitem__(self,idx):

		return self.data[idx]


	def update_data(
		self,
		delta:float = None,
		tau:float = None
		)->None:


		r"""
			A function to update the stored data list. When the loader is first initialized this function is called with the
			values of delta and tau that were passed to it. Subsequent calls check if the passed values are empty (and thus no update)
			and allow only one of the parameters to be updated. 
		"""
		# print(f"Calling update data with delta={delta} and tau = {tau}")
		# time.sleep(2)

		if delta is None and tau is None: 
			# nothing to update
			# print("not updating")
			return



		if delta is not None and tau is None:
			# print("only updating delta")
			if delta == self.delta:
				return
			
			new_tau = self.tau 
			new_delta = delta
			# print(new_tau)
			# print(new_delta)
			# time.sleep(3)
	
		elif delta is None and tau is not None:
			# print("only updating tau")
			if tau == self.tau:
				return
			new_tau = tau
			new_delta = self.delta
			# print(new_tau)
			# print(new_delta)
			# time.sleep(3)

		else:
			# print("updating both")
			new_tau = tau
			new_delta = delta
			# print(new_tau)
			# print(new_delta)
			# time.sleep(3)

		# find the index above which all the annotated data points have a gap
		# at least as large as new_delta

		split_idx = bisect.bisect_left(self.gap_list, new_delta)
		
		# the subset is those annotated data points at or beyond that index
		data_subset = self.ann_datapoints[split_idx:]
		
		self.data = [
			(data_tuple[0],data_tuple[1],data_tuple[2]) 
			for data_tuple in data_subset 
			if data_tuple[4]>= new_tau]

		data_subset = [data_subset[j] for j in range(len(data_subset)) if data_subset[j][3]>=new_tau]
		
		
		self.delta = new_delta
		self.tau = new_tau
			


	

	def _build_gap_lists(self) ->None:
		
		r"""
		
		A utility function that allows for speedy updating of the data for new values of 
		beta and tau.

		Instead of creating a new instance of a data loader for each possible pair of beta and tau values
		we store the list of gaps and the largest value.

		The loader keeps the provided tensor of omics values (X) and response measures (y). This function adds
		two new data structures:

			- A collection of annotated data points which are of the form (x1, x2, target, max, gap) where x1,x2 are
			  omics measurements, the target is the value to be passed to the loss function, max is the larger of the
			  response values for samples x1 and x2 (max(y[1],y[2])) and gap is the absolute value of 
			  the difference between these two (|y[1]-y[2]|).
			- A gap list which is the list of the gap values of each data point. 


		We store the annotated data points and the gap list in increasing order of the size of the gap.
		Using the bisect method this allows for rapid subsetting of the data during update.

		"""

		N = self.y.shape[0]

		# iterator to go through all ordered pairs of observations.
		if self.symmetric_pairs:
			data_iterator = permutations(range(N),2)
		else:
			data_iterator = combinations(range(N),2)


		# create a list of annotatd data points which are tuples of the form
		# (X[i,], X[j,], y_{ij}, max(y[i], y[j]), |y[i]-y[j]|)
		ann_datapoints = [ 
			(self.X[i,:],self.X[j,:],self._compute_target(i,j), self._compute_max_resp(i,j), self._compute_gap(i,j),
				self.y[i],self.y[j]) 
			for i,j in data_iterator
			]

	
		# sort the pairs in increasing order of the size of their gaps
		ann_datapoints = sorted(ann_datapoints, key = lambda x: x[4])
		
		# keep track of the gaps themselves
		gap_list = [ann_point[4] for ann_point in ann_datapoints]
		
		self.ann_datapoints = ann_datapoints
		self.gap_list = gap_list
		
		

	def _compute_gap(self,
		i:int,
		j:int
		)->float:
		r"""
		A function to compute the gap betweeen two drug response measurements.


		Args:

			i: int - The first index of values to be checked
			j: int - The second index of to be checked


		return:
			The absolute value of the difference |y[i]-y[j]|
		"""
		return float(abs(self.y[i]-self.y[j]))


	def _compute_max_resp(self,
		i:int, 
		j:int
		)->float:
		
		r"""

		A function to compute the largest response value between two indices i and j.

		Args:

			i: int - The index of the first sample to be checked.
			j: int - The index of the second sample to be checked. 

		
		return:
			The maximum of the response values y[i], y[j]
		"""
		return float(max(self.y[i],self.y[j]))

	def _compute_target(self,
		i:int,
		j:int
		)->torch.tensor:
		r"""

		Function that computes the target value for passing to a loss function. 


		Args:
			i: int - index of the first sample 
			j: int - index of the second sample


		return:
			+/- 1 if using a margin ranking loss and 0/1 if using a probability loss.
		"""

		if self.prob_loss:
			target = 1 if self.y[i] >= self.y[j] else 0	
		else:
			target = 1 if self.y[i]>self.y[j] else -1


		target = torch.tensor(target,dtype=torch.float32)
		return target

	





