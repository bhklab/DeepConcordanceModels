import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import os
import utils
import evaluation_metrics as evm
from scipy.stats import pearsonr
from collections import defaultdict



def find_best_delta(nm,ds):
	df = pd.read_csv(f"../results/{drug}/neurals/{ds}/train/{nm}.csv",index_col=0)
	
	deltas = pd.unique(df['Delta'])

	best_corr = 0
	best_ci = 0
	ci_delta = -1
	corr_delta = -1

	for delta in deltas:
		_temp = df[df['Delta']==delta]
		corr,_ = pearsonr(_temp['Predicted'],_temp['AAC'])
		# ci = evm.robust_concordance(_temp['Predicted'].values,_temp['AAC'].values)
		if corr>best_corr:
			best_corr = corr
			corr_delta = delta
		
		# if ci>best_ci:
		# 	best_ci = ci
		# 	ci_delta = delta
		# 	results['Delta'].append(delta)
		# results['Pearson'].append(corr)
		# results['CI'].append(evm.robust_concordance(_temp['Predicted'],_temp['AAC']))
	
	return corr_delta, best_corr, ci_delta, best_ci


drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")

model_name_map = {'ECINET':"ECINET",'ONE_NARROW':"One Layer", "TWO_NARROW":"Two Layer"}

for ds in ['replicate_gaps','param_sweep']:
	
	delta_res = defaultdict(list)
	
	for drug in drugs:
		best_deltas = defaultdict(float)
		nm_res = defaultdict(list)
		for nm in model_name_map:
			corr_d,best_corr,ci_d,best_ci =  find_best_delta(nm,ds)
			delta_res['Drug'].append(drug)
			delta_res['Model'].append(model_name_map[nm])
			delta_res['Delta'].append(corr_d)
			delta_res['Pearson'].append(best_corr)
			# delta_res['Delta.C'].append(ci_d)
			# delta_res['CI'].append(best_ci)
	delta_res = pd.DataFrame(delta_res)
	delta_res.to_csv(f"../results/{ds}_best_deltas.csv")

