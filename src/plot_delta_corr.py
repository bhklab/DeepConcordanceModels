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
	best_delta = -1

	for delta in deltas:
		_temp = df[df['Delta']==delta]
		corr,_ = pearsonr(_temp['Predicted'],_temp['AAC'])
		
		if corr>best_corr:
			best_corr = corr
			best_delta = delta
		# 	results['Delta'].append(delta)
		# results['Pearson'].append(corr)
		# results['CI'].append(evm.robust_concordance(_temp['Predicted'],_temp['AAC']))
	
	return best_delta, best_corr


delta_source = "replicate_gaps"

model_name_map = {'ECINET':"ECINET",'ONE_NARROW':"One Layer", "TWO_NARROW":"Two Layer"}

gaps = pd.read_csv("../results/Broad/empirical_deltas.csv",index_col=0)
gaps = gaps[gaps['Type']=='All']
deltas = {k:v for k,v in zip([0]+list(gaps['Percentile']),[0]+list(gaps['Delta']))}

screens = ['gCSI','GDSC2']
drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")
ep = 30

baseline_models = ['ElasticNet','RandomForest','Univariable']

drugs = drugs[1:4]
all_res = []
delta_res = defaultdict(list)
for drug in drugs:
	best_deltas = defaultdict(float)
	nm_res = defaultdict(list)
	for nm in model_name_map:
		d,c = find_best_delta(nm,delta_source)
		delta_res['Drug'].append(drug)
		delta_res['Model'].append(nm)
		delta_res['Delta'].append(d)
		delta_res['Train Pearson'].append(c)
		n_pool = []
		
		for screen in screens:
			n_res = pd.read_csv(f"../results/{drug}/neurals/{delta_source}/test/{screen}/{nm}.csv",index_col=0)
			n_res = n_res[n_res['Delta']==d]
			n_pool.append(n_res)
			corr, _ = pearsonr(n_res['Predicted'],n_res['AAC'])
			ci = evm.robust_concordance(n_res['Predicted'].values,n_res['AAC'].values)
			nm_res['Model'].append(model_name_map[nm])
			nm_res['Screen'].append(screen)
			nm_res['Concordance'].append(ci)
			nm_res['Pearson'].append(corr)
		n_res = pd.concat(n_pool)
		corr, _ = pearsonr(n_res['Predicted'],n_res['AAC'])
		
		ci = evm.robust_concordance(n_res['Predicted'].values,n_res['AAC'].values)
		nm_res['Model'].append(model_name_map[nm])
		nm_res['Screen'].append("Pooled")
		nm_res['Concordance'].append(ci)
		nm_res['Pearson'].append(corr)
	nm_res = pd.DataFrame(nm_res)
	nm_res['Drug']=drug
	bm_res = defaultdict(list)
	for bm in baseline_models:
		model_pool = []	

		for screen in screens:
			baseline_res = pd.read_csv(f"../results/{drug}/baselines/test/{screen}/{bm}.csv",index_col=0)
			# print(baseline_res)
			corr, _ = pearsonr(baseline_res['Predicted'],baseline_res['AAC'])
			ci = evm.robust_concordance(baseline_res['Predicted'].values,baseline_res['AAC'].values)

			bm_res['Model'].append(bm)
			bm_res['Screen'].append(screen)
			bm_res['Concordance'].append(ci)
			bm_res['Pearson'].append(corr)
			model_pool.append(baseline_res)
		
		model_res = pd.concat(model_pool)
		corr, _ = pearsonr(model_res['Predicted'],model_res['AAC'])
		
		ci = evm.robust_concordance(model_res['Predicted'].values,model_res['AAC'].values)
		bm_res['Model'].append(bm)
		bm_res['Screen'].append("Pooled")
		bm_res['Concordance'].append(ci)
		bm_res['Pearson'].append(corr)
	# print(bm)
	# print(corr)
		# print(best_deltas)
	bm_res = pd.DataFrame(bm_res)
	bm_res['Drug']=drug
	ar = pd.concat([nm_res,bm_res])
	all_res.append(ar)
delta_res = pd.DataFrame(delta_res)
print(delta_res)
all_res = pd.concat(all_res)
all_res.to_csv(f"../results/{delta_source}_scalars.csv")
	# for screen in screens:
	# 	baselines = []
	# 	for bm in baseline_models:

