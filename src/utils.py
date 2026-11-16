import numpy as np
from typing import List,Dict,Tuple, Union
import pandas as pd
from collections import defaultdict
import evaluation_metrics as evm

from itertools import chain, combinations
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoCV, Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold, PredefinedSplit
from sklearn.metrics import root_mean_squared_error, ndcg_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from lifelines.utils import concordance_index
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, label_ranking_loss, label_ranking_average_precision_score, precision_score, recall_score
from evaluation_metrics import rCI_score

SKLEARN_MODELS = ["ElasticNet","Lasso","Ridge","KRR","RandomForest","Univariable", "Dummy"]


def make_ordinal(
	y:np.ndarray,
	cutoff:float = 0.,
	output_type:str = 'binary'
	)->np.ndarray:

	r"""
	
	Some of our metrics are only computable from
	binary or ordinal relevance scores.

	This function takes in a vector of non-negative scores and
	either binarizes based on a passed in cutoff or
	returns 


	"""
	
	if output_type=="binary":
		ordinal_y = np.where(y >= cutoff,1,0)
	elif output_type == "scalar":
		ordinal_y = [int(a) for a in np.round(100*y)]

	return ordinal_y



def make_threshold_map(
	values:np.array,
	thresholds:List[int]
	) ->Dict:
	thresh_to_cutoff = {}
	
	for thresh in thresholds:
		cutoff = np.round(np.percentile(values,100-thresh),4)
		thresh_to_cutoff[thresh] = float(cutoff)

	return thresh_to_cutoff


def unpack_parameters(
    D:Dict,
    return_dict:bool=False
    ):
	
	if len(D.values())>1:
		return D if return_dict else tuple(D.values())
	else:
		return D if return_dict else tuple(D.values())[0]



def read_text_file(
	fpath:str
	) -> List[str]:
	
	with open(fpath,"r") as istream:
		lines = istream.readlines()
	lines = [x.rstrip() for x in lines]
	return lines

def fetch_and_subset_data(
	expression_file:str,
	response_file:str,
	ccl_col:str,
	cell_lines:List[str],
	genes:List[str]
	)-> Tuple[pd.DataFrame,pd.DataFrame]:
	"""
	
	Utility function that reads in the gene expression and response data, 
	subsets the expression to only be the genes of interest and the measurements
	only to be from cell lines we care about (e.g., those from solid tumors).


	"""
	expr = pd.read_csv(expression_file)
	resp = pd.read_csv(response_file)
	expr = expr[[ccl_col]+[gene for gene in genes if gene in expr.columns]]
	resp = resp[resp[ccl_col].isin(cell_lines)]
	expr = expr[expr[ccl_col].isin(cell_lines)]

	return expr,resp
	



def build_data_set(
	data: pd.DataFrame,
	genes:List[str],
	ccl_col:str = "Cell Line",
	viability_col:str = 'AAC',
	ccl_subset:List[str] = None
	)->Tuple[np.array,np.array]:
	
	if ccl_subset is not None:
		_data = data[data[ccl_col].isin(ccl_subset)]
	else:
		_data = data

	y = _data[viability_col].values.reshape(-1)
	

	expr = _data[[gene for gene in genes if gene in _data.columns]]
	X = expr.values

	return X, y
		



def compute_scalar_metrics(
	res_dict:Dict,
	drug:str, 
	data_set:str,
	model_name:str,
	eval_type:str,
	y_true:np.array,
	preds:np.array,
	gap_range:np.array,
	delta:float = -1,
	cv_score:str = None
	)->None:
	

	# ci = float(concordance_index(y_true,preds))
	ci = evm.robust_concordance(y_true,preds)
	
	
	if np.allclose(preds,preds[0]):
		corr = 0
	else:
		corr,_ = pearsonr(y_true,preds)  # res_dict['RMSE'].append(rmse)
	
	for gap in gap_range:

		
		rCI_both = evm.robust_concordance(y_true,preds,gap,gap)
		rCI_pred = evm.robust_concordance(y_true,preds,gap,0)
	

		

		res_dict['Drug'].append(drug)
		res_dict['Data Set'].append(data_set)
		res_dict['Eval. Type'].append(eval_type)
		res_dict['Model'].append(model_name)
		res_dict["Pearson"].append(corr)
		res_dict['Concordance'].append(ci)
		res_dict['rCI'].append(rCI_both)
		res_dict['rCI_preds'].append(rCI_pred)
		res_dict['Gap'].append(gap)
		res_dict['Delta'].append(delta)
		res_dict['cv_score'].append(str(cv_score))

		

def _get_tens_digits(
	in_arr: np.array
	) -> np.array:

	out_arr = np.array([int(str(x).split(".")[1][0]) if x<1. else 10 for x in in_arr ])
	return out_arr

def compute_ranking_metrics(
	res_dict:Dict,
	drug:str, 
	data_set:str,
	model_name:str,
	eval_type:str,
	y_true:np.array,
	preds:np.array,
	k_range:np.array,
	delta:float = -1,
	thresh:int = None,
	cutoff:float = None,
	cv_score:str = None):
	

	
	_k_range = [x for x in k_range if x<y_true.shape[0]]
	y_true_tens = _get_tens_digits(y_true)
	y_true_bin = make_ordinal(y_true,cutoff)
	order = np.argsort(-1*preds)
	y_true_bin_sorted = y_true[order]
	lrl = label_ranking_loss(y_true_bin.reshape(1,-1), preds.reshape(1,-1))
	
	lrap = label_ranking_average_precision_score(y_true_bin.reshape(1,-1), preds.reshape(1,-1))
	
	rr = evm.reciprocal_rank(y_true_bin,preds)
	num_rel = np.sum(y_true_bin)
	for k in _k_range:
		# scalar_results['Gene Set'].append(gs)		
		perm_prec = evm.perm_overlap(y_true,preds,k=k)
		# ndcg0 = evm.ndcg_at_k(y_true, preds, k, method = 0)
		# ndcg1 = evm.ndcg_at_k(y_true,preds,k, method=1)
		ndcg  = ndcg_score(y_true.reshape(1,-1),preds.reshape(1,-1),k=k)
		ndcg_ordinal = ndcg_score(y_true_tens.reshape(1,-1),preds.reshape(1,-1),k=k)
		ndcg_binary = ndcg_score(y_true_bin.reshape(1,-1),preds.reshape(1,-1),k=k)
		

		ap_k = evm.average_precision_at_k(y_true_bin,preds,k)
		prec_at_k = np.sum(y_true_bin_sorted[:k])/k
		recall_at_k = np.sum(y_true_bin_sorted[:k])/num_rel if num_rel >0 else -1
		res_dict['Drug'].append(drug)
		res_dict['Data Set'].append(data_set)
		res_dict['Eval. Type'].append(eval_type)
		res_dict['Model'].append(model_name)
		res_dict['Perm Overlap'].append(perm_prec)
		res_dict['NDCG'].append(ndcg)
		res_dict['Ordinal NDCG'].append(ndcg_ordinal)
		res_dict['ndcg_binary'].append(ndcg_binary)
		res_dict['Recip Rank'].append(rr)
		res_dict['Label Ranking Loss'].append(lrl)
		res_dict['Label Ranking Average Precision'].append(lrap)
		res_dict['AP@K'].append(ap_k)
		res_dict['P@K'].append(prec_at_k)
		res_dict['R@K'].append(recall_at_k)
		res_dict['K'].append(k)
		res_dict['Delta'].append(delta)
		res_dict['Threshold'].append(thresh)
		res_dict['Cutoff'].append(cutoff)
		res_dict['cv_score'].append(str(cv_score))




def fetch_partitioned_data(
	train_df:pd.DataFrame, 
	train_ccls:List[str], 
	testing_screens:List[str],
	gene_set:str,
	drug:str, 
	ccl_type:str = 'solid'
	)->Dict[str,Dict[str,pd.DataFrame]]:
	
	partitioned_test_data = {}
	
	for test_screen in testing_screens:

		partitioned_test_data[test_screen] = {}
		test_df = pd.read_csv(f"../procdata/{test_screen}/{ccl_type}/{gene_set}/{drug}/data.csv")
		test_ccls = pd.unique(test_df[test_df.columns[0]])
		


		# for analyzing the ground truth we keep the overlapping columns and merge
		# to make sure they're in the right order
		ccl_col, viability_col = train_df.columns[0], train_df.columns[-1]
		overlap_df = train_df[[ccl_col,viability_col]].merge(test_df[[ccl_col,viability_col]],on =ccl_col)
		

		# keep the values for later so that it follows the model syntax
		preds_gt = overlap_df[f"{viability_col}_x"].values
		y_test_gt = overlap_df[f"{viability_col}_y"].values
		
		seen_ccls = [x for x in test_ccls if x in train_ccls]
		unseen_ccls = [x for x in test_ccls if x not in train_ccls]

		for eval_type, ccl_subset in zip(
			['Seen','Unseen','All'], 
			[seen_ccls,unseen_ccls,None]
			):
			


			if ccl_subset is not None:
				data_subset = test_df[test_df[test_df.columns[0]].isin(ccl_subset)]

			X_test = data_subset[data_subset.columns[1:-1]].values
			y_test = data_subset[data_subset.columns[-1]].values

			if eval_type == "Seen":
				partitioned_test_data[test_screen][eval_type] = (X_test,y_test,preds_gt,y_test_gt)
			else:
				partitioned_test_data[test_screen][eval_type] = (X_test,y_test)

	return partitioned_test_data


def build_model(
	model_name:str,
	alpha_min:float = None,
	alpha_max:float = None,
	num_alphas:int = None,
	alpha_logspace:bool=True,
	l1_ratios:np.array = None,
	cv_obj:Union[int,PredefinedSplit] = 5,
	max_iter:float = 10**4,
	precompute:Union[bool,str] = "auto",
	fit_intercept:bool = False,
	selection:str = 'random',
	max_features:List[Union[int,str]] = [1,2,3],
	n_estimators:List[int] = [10],
	n_jobs:int = -1,
	k:int = 1,
	cv_score:str = None,
	random_state = None
	):
	
	if alpha_min is not None:
		if alpha_logspace:
			alphas = np.logspace(alpha_min,alpha_max,num_alphas)
		else:
			alphas = np.linspace(alpha_min,alpha_max,num_alphas)



	if cv_score =="rCI":
		scorer = make_scorer(score_func = rCI_score)
	elif cv_score == None:
		scorer = None


		
	if model_name=="ElasticNet":
		

		pipe = Pipeline([('scale',StandardScaler()),
				   ('regr',ElasticNet(max_iter=max_iter, selection = 'random',random_state=random_state))])
		param_grid = {
		'regr__alpha': alphas,
		'regr__l1_ratio': l1_ratios,
		'regr__fit_intercept':[True,False]
		}

		model = GridSearchCV(pipe,param_grid,scoring = scorer,n_jobs= n_jobs,cv = cv_obj)
	

		# model = ElasticNetCV(
		# 				l1_ratio = l1_ratios,
		# 				alphas = alphas,
		# 				cv = cv_obj,
		# 				precompute = precompute,
		# 				max_iter = max_iter,
		# 				selection = 'random',
		# 				n_jobs = n_jobs,
		# 				fit_intercept = fit_intercept)
	
	elif model_name == "Lasso":
		
		
		pipe = Pipeline([('scale',StandardScaler()),
				   ('regr',Lasso(max_iter=max_iter, random_state=random_state))])
		param_grid = {
		'regr__alpha': alphas,
		'regr__fit_intercept':[True,False]
		}


		model = GridSearchCV(pipe,param_grid,scoring = scorer, n_jobs= n_jobs,cv = cv_obj)





		# model = LassoCV(
		# 				alphas = alphas, 
		# 				cv = cv_obj,
		# 				max_iter = max_iter,
		# 				n_jobs = n_jobs,
		# 				fit_intercept = fit_intercept)
	
	elif model_name == "Ridge":
		

		pipe = Pipeline([('scale',StandardScaler()),
				   ('regr',Ridge(max_iter=max_iter,random_state=random_state))])
		param_grid = {
			'regr__alpha': alphas,
			'regr__fit_intercept':[True,False]
		}

		model = GridSearchCV(pipe,param_grid,scoring = scorer,n_jobs= n_jobs,cv = cv_obj)
		# model = RidgeCV(alphas = alphas, 
		# 				cv = cv_obj)

	elif model_name == "RandomForest":
		regr = RandomForestRegressor(n_jobs = n_jobs,random_state=random_state)
		param_grid = {
			'max_features': max_features,
			'n_estimators': n_estimators
			}
		model = GridSearchCV(regr,param_grid,scoring = scorer,n_jobs= n_jobs,cv = cv_obj)

	elif model_name == "KRR":
		
		regr = KernelRidge()
		param_grid = {'alpha':alphas}
		model = GridSearchCV(regr,param_grid,scoring = scorer,n_jobs=-1, cv = cv_obj)
		
	
	elif model_name == "Univariable":
		
		model = Pipeline([
			('select',SelectKBest(mutual_info_regression,k=k)),
			('predict',LinearRegression())
			])

	elif model_name == "Dummy":

		model = DummyRegressor()

	return model


if __name__ == '__main__':
	rng = np.random.default_rng(seed=1234)
	samples = rng.normal(0,1,100)
	print(samples)
	thresholds = [1,5,10,20]

	tm = make_threshold_map(samples,thresholds)
	# print(tm)
	for t in tm.keys():
		print(tm[t])
		p = make_ordinal(samples,tm[t])
		print(np.sum(p))