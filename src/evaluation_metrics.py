from lifelines.utils import concordance_index
from typing import List, Union, Tuple
""""
Metrics for evaluating ranking models.

These functions are designed to mirror the structure
of the scikit-learn evaluation metrics. That is, in general
they take the form of:
	
	ranking_metric(y_true, y_est, **kwargs)

where:
	- y_true are the true relevance judgements,
	- y_est are the estimated scores 


All of these metrics presume that higher scores represent a 
better match/better relevance. 

Author: James Bannon
Github: jbannon


For future work:
	- Expected Reciprocal Rank
	- R-Precision

"""
import numpy as np
from typing import Union, List




"""
Metrics for Binary Relevance Judgements
"""

def reciprocal_rank(
	y_true:Union[np.array,List[int]],
	y_est:Union[np.array,List[float]]
	)->float:
	

	r"""
	Compute the reciprocal rank of an estimated sorted ranking score.
	If rels


	Parameters
	----------

	y_true: np.array
		The collection of actual relevance judgements. These must be binary (0/1)

	y_est: np.array
		Real valued scores of the documents by a LETOR system. Higher score is better. 

	""" 
	assert all([score in [0.,1.] for score in y_true ]), "y_true must contain only binary relevance judgements"

	y_true = np.asarray(y_true)
	y_est = np.asarray(y_est)
	
	order = np.argsort(-1*y_est)
	

	rank = np.asarray(y_true[order]).nonzero()[0]
	
	recip_rank = 1/(float(rank[0])+1) if rank.size else 0.
	
	return recip_rank


def precision_at_k(
	y_true:Union[np.array,List[int]],
	y_est:Union[np.array, List[float]],
	k:int = 5,
	) -> float:

	r"""

	Compute the precision at k for a given value of k

	Parameters
	----------

		y_true: np.array
			A binary vector with 1/0 relevance judgements
		y_est: np.array
			An array of real valued scores
		k: int
			The cutoff point for considering precision

	"""
	assert k>=1, "k must be greater than or equal to 1"
	
	y_est = np.asarray(y_est)
	y_true = np.asarray(y_true)

	order = np.argsort(-1*y_est)
	ordered_truth = y_true[order]

	return (1.*np.sum(ordered_truth[:k])/k)

def average_precision_at_k(
	y_true:Union[np.array,List[int]],
	y_est:Union[np.array,List[float]],
	k:int = 5
	)->float:
	r"""

	A function to compute the average precision
	
	
	"""


	assert k>=1, "k must be greater than or equal to 1"


	y_true = np.asarray(y_true)
	y_est = np.asarray(y_est)
	
	order = np.argsort(-1*y_est)
	y_sorted = y_true[order]
	

	prec_values = np.array(
		[precision_at_k(y_true,y_est,k+1) for k in range(k) if y_sorted[k]==1]
		)
	if not prec_values.size>=1:
		ap=0
	else:
		ap = np.mean(prec_values)
	return ap




"""
Metrics for Real-Valued Relevance Judgements
"""
def perm_overlap(
	y_true:Union[np.array,List[float]],
	y_est:Union[np.array,List[float]],
	k:int = 5
	)-> float:
	
	r"""
		
	This function computes a permutation based precision score. 
	
	In particular, given two arrays, one $y$ of true scores
	(which can be any real value) and another $f$ of scores 
	produced by a model, this metric measures how many of the top k
	items in the true ordering are in the top k in the model score
	ordering.

	Suppose $\pi$ sorts $f$ in decreasing order and $\sigma$ sorts 
	$y$ in decreasing order. This function gives

	..math::
		\frac{1}{k}\sum_{i=1}^k\sum_{j=1}^k \delta(\pi_i,\sigma_j)

	where $\delta$ is the kronecker delta.


	Parameters
	----------

	y_true: np.array
		An array of the true relevance scores

	y_est: np.array
		An array of scores produced by a model

	k: int
		The cutoff for computing the score. 


	Notes
	-----

	This metric is used in the paper
		"Kernelized rank learning for personalized drug recommendation" 
			by He, Folkman, & Borgwardt

	and is defined in equation 12 in that paper. 

	""" 
	assert k>=1, "k must be greater than 1"

	y_true = np.asarray(y_true)
	y_est = np.asarray(y_est)

	true_order = np.argsort(-y_true)
	est_order = np.argsort(-y_est)

	overlap_size = 0
	
	for i in range(k):
		for j in range(k):
			if est_order[i]==true_order[j]:
				overlap_size+=1

	return 1.*overlap_size/k
	
def dcg_at_k(
	rel_scores:np.ndarray,
	k:int = 5,
	method:int = 1
	)->float:
	r"""
	
	A function to compute the discounted cumulative gain of a numpy array x. It allows the
	user to specify whether or not they want to use the original definition of cumulative gain as
	defined in [1] or the more stringently penalized formulation in [2]. The default
	behavior is [2] as that formulation is more widely used. 

	

	Parameters
	----------
		
		rel_scores (np.array): 
			An array of ordinal scores. Assumes that higher score is better. 
		
		k (int): 
			An integer specifying how many positions to look at. Look at only the first k positions. 
			(default: :obj: 5)
		
		method (int):
			An integer specifying which dcg computation to use; method = 0 corresponds to [1] 
			and method = 1 corresponds to [2].
			(default: :obj: 1)


	

	Notes
	-----


	[1] Järvelin, Kalervo, and Jaana Kekäläinen. "Cumulated gain-based evaluation of IR techniques."
		ACM Transactions on Information Systems (TOIS) 20.4 (2002): 422-446.
	
	[2] Burges, Chris, et al. "Learning to rank using gradient descent."
		Proceedings of the 22nd international conference on Machine learning. 2005.

	"""

	rel_scores = np.asarray(rel_scores)

	assert np.all(rel_scores>=0), "relevance judgements must all be non-negative"
	assert k>=1, "k must be greater than or equal to 1"
	assert method in [0,1], "method must be either 0 or 1"
	

	rel_scores = np.asarray(rel_scores)[:k]


	

	if method == 1: 
		
		dcg =  np.sum((np.exp2(rel_scores)-1)/np.log2([i+1 for i in range(1,rel_scores.size+1)]))
	
	else:

		dcg = rel_scores[0] + np.sum(rel_scores[1:]/np.log2(np.arange(2,rel_scores.size+1)))
	
	return dcg


def ndcg_at_k(
	y_true:np.array, 
	y_est:np.array,
	k:int = 5,
	method:int = 1
	)->float:
	
	r""" 
	Compute the normalized discounted cumulative gain (NDCG) at position k. 
	 

	We assume that higher scores are better. 

	Args:
		y_true (np.array): A vector of true relevance judgements.
		document
		
		y_est (np.array): A vector of estimated scores.
		
		k (int): how far down the list to check

	"""	

	assert k>=1, "k must be greater than or equal to 1"
	y_est = np.asarray(y_est)
	y_true = np.asarray(y_true)
	est_order = np.argsort(-1*y_est)
	opt_order = np.argsort(-1*y_true)



	est_dcg = dcg_at_k(y_true[est_order],k,method)
	opt_dcg = dcg_at_k(y_true[opt_order],k,method)
	
	# print(est_dcg)
	ndcg = est_dcg/opt_dcg
	
	return ndcg



def robust_concordance(
	observed:np.array,
	predicted:np.array,
	delta_p:float = 0,
	delta_o:float = 0,
	ignore_ties:bool = True, ## outx
	allow_equality:bool = True ## logic operator
	):
	


	assert len(predicted) == len(observed), "arrays must have same length"
	N = len(predicted)

	c, d, u  = np.zeros((1,N)), np.zeros((1,N)),np.zeros((1,N))

	count_comparable = 0
	for i in range(N-1):
		for j in range(i+1,N):
			pair = [i,j]
			
			pair_is_comparable = np.abs(observed[i]-observed[j])>delta_o and np.abs(predicted[i]-predicted[j])>delta_p
			if pair_is_comparable:
				count_comparable+=1
			if allow_equality:
				is_edge_case = abs(predicted[i]-predicted[j])==delta_p
			else:
				is_edge_case = not pair_is_comparable


			if pair_is_comparable:
				pp = predicted[i]>predicted[j]
				oo = observed[i]>observed[j]
				
				if pp == oo:	
					c[:,pair]+=1
				else:
					d[:,pair]+=1
			elif is_edge_case:
				if ignore_ties or np.abs(observed[i]-observed[j])<=delta_o:
					u[:,pair]+=1
				else:
					c[:,pair]+=0.5
					d[:,pair]+=0.5

	C = np.sum(c).item()
	D = np.sum(d).item()
	if C+D ==0:
		CI = -1
	else:
		CI = C/(C+D)
	
	return CI



def random_concordance(
	observed:np.array,
	predicted:np.array,
	seed:int = 1234,
	threshold:float = 0.2,
	ignore_ties:bool = True, ## outx
	delta_p:float = 0,
	delta_o:float = 0,
	allow_equality:bool = True ## logic operator
	):
	

	rng = np.random.default_rng(seed)
	assert len(predicted) == len(observed), "arrays must have same length"
	N = len(predicted)

	c, d, u  = np.zeros((1,N)), np.zeros((1,N)),np.zeros((1,N))

	count_comparable = 0
	for i in range(N-1):
		for j in range(i+1,N):
			pair = [i,j]
			
			pair_is_comparable = rng.uniform()<=threshold
			if pair_is_comparable:
				count_comparable+=1
			if allow_equality:
				is_edge_case = abs(predicted[i]-predicted[j])==delta_p
			else:
				is_edge_case = not pair_is_comparable


			if pair_is_comparable:
				pp = predicted[i]>predicted[j]
				oo = observed[i]>observed[j]
				
				if pp == oo:	
					c[:,pair]+=1
				else:
					d[:,pair]+=1
			elif is_edge_case:
				if ignore_ties or np.abs(observed[i]-observed[j])<=delta_o:
					u[:,pair]+=1
				else:
					c[:,pair]+=0.5
					d[:,pair]+=0.5

	C = np.sum(c).item()
	D = np.sum(d).item()
	if C+D ==0:
		CI = -1
	else:
		CI = C/(C+D)
	
	return CI

def rCI_score(
	y_true:np.array,
	y_pred:np.array,
	delta_p:float = 0,
	delta_o:float = 0,
	ignore_ties:bool = True, ## outx
	allow_equality:bool = True ## logic operator
	):
	


	assert len(y_true) == len(y_pred), "arrays must have same length"
	N = len(y_pred)

	c, d, u  = np.zeros((1,N)), np.zeros((1,N)),np.zeros((1,N))

	
	for i in range(N-1):
		for j in range(i+1,N):
			pair = [i,j]
			
			pair_is_comparable = np.abs(y_true[i]-y_true[j])>delta_o and np.abs(y_pred[i]-y_pred[j])>delta_p

			if allow_equality:
				is_edge_case = abs(y_pred[i]-y_pred[j])==delta_p
			else:
				is_edge_case = not pair_is_comparable


			if pair_is_comparable:
				pp = y_pred[i]>y_pred[j]
				oo = y_true[i]>y_true[j]
				
				if pp == oo:	
					c[:,pair]+=1
				else:
					d[:,pair]+=1
			elif is_edge_case:
				if ignore_ties or np.abs(observed[i]-observed[j])<=delta_o:
					u[:,pair]+=1
				else:
					c[:,pair]+=0.5
					d[:,pair]+=0.5

	C = np.sum(c).item()
	D = np.sum(d).item()
	if C+D ==0:
		CI = -1
	else:
		CI = C/(C+D)
	return CI


def spearman_distance(
	y_true:Union[List,np.array],
	y_pred:Union[List,np.array]
	)->Tuple[float,float]:

	assert len(y_true) == len(y_pred), "Passed Arrays must have same length"

	N = len(y_true)

	true_order = np.argsort(y_true)+1
	pred_order = np.argsort(y_pred)+1

	distance = np.sum(np.abs(true_order-pred_order))
	scaled_distance = distance/N

	return distance, scaled_distance




def my_concordance(
	observed:np.array,
	predicted:np.array,
	)-> float:

	assert len(predicted)==len(observed), "arrays must have same length"
	N = len(predicted)
	count = 0
	scaling_factor = 2./(N*(N-1))

	for i in range(N-1):
		for j in range(1,N):
			if i !=j:
				print(f"({i},{j}) - ")
				print(f"Preds: ({predicted[i]},{predicted[j]})")
				print(f"Est: ({observed[i]},{observed[j]})")
				if predicted[i]>predicted[j] and observed[i]>observed[j]:
					print("incr. count")
					count+=1

	return scaling_factor*count

def main():
	print("---- running unit tests -----")
	seed = 1234
	y_cont = [0.2, 0.8, 0.5, 0.6, 0.1, 0.75, 0.9]
	y_bin = [0, 1,0,1,0,1,0]
	y_est = [0.3, 0.5, 0.1, 0.65, 0.7, 0.4, 0.2]


	print(my_concordance(y_cont,y_est))
	print(concordance_index(y_cont,y_est))
	print(robust_concordance(y_cont,y_est))
	print(robust_concordance(y_cont,y_est,delta_o=0.2,delta_p=0))
	print(robust_concordance(y_cont,y_est,delta_o=0.2,delta_p=0.2))
	# d,s = spearman_distance(y_cont, y_est)
	# print(d)
	# print(s)
	# print(perm_overlap(y_cont,y_est))
	# print("testing reciprocal rank")
	# print(reciprocal_rank(y_bin, y_est))


	# print("testing precision at k")
	# for k in range(1,7):
	# 	print(precision_at_k(y_bin, y_est,k))
	

	# print("testing average precision at k")
	# for k in range(1,7):
	# 	print(average_precision_at_k(y_bin, y_est,k))
	

	# print("Testing dcg")	
	
	# r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]

	# print(dcg_at_k(r,k=1))
	# print(dcg_at_k(r, 1, method=1))
	# print(dcg_at_k(r, 2))

	# print(dcg_at_k(r, 2, method=1))
	
	# print(dcg_at_k(r, 10))
	# print(dcg_at_k(r, 11))

	# print("testing ndcg")
	# for k in range(1,7):
	# 	for method in [0,1]:
	# 		print(ndcg_at_k(y_cont, y_est, k,method))

	
	# print("testing perm precision")

	# for k in range(1,8):
	# 	print(k)
	# 	print(perm_precision(y_cont, y_est,k))



if __name__ == '__main__':
	main()