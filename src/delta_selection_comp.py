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


screens = ['gCSI','GDSC2']
baseline_models = ['ElasticNet','RandomForest','Univariable']

src = "replicate_gaps"
drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")
model_name_map = {'ECINET':"ECINET",'ONE_NARROW':"One Layer", "TWO_NARROW":"Two Layer"}


bd = pd.read_csv(f"../results/{src}_best_deltas.csv",index_col = 0)

all_res = []
gaps = defaultdict(list)
for drug in drugs:
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
	
	nm_res = defaultdict(list)
	for nm in model_name_map:
		model_pool = []
		d = bd[ (bd['Drug']==drug) & (bd['Model']==model_name_map[nm])]['Delta'].values[0]
		
		for screen in screens:
			n_res = pd.read_csv(f"../results/{drug}/neurals/{src}/test/{screen}/{nm}.csv",index_col=0)
			n_res = n_res[n_res['Delta']==d]
			model_pool.append(n_res)
			corr, _ = pearsonr(n_res['Predicted'],n_res['AAC'])
			ci = evm.robust_concordance(n_res['Predicted'].values,n_res['AAC'].values)
			nm_res['Model'].append(model_name_map[nm])
			nm_res['Screen'].append(screen)
			nm_res['Concordance'].append(ci)
			nm_res['Pearson'].append(corr)
		n_res = pd.concat(model_pool)
		corr, _ = pearsonr(n_res['Predicted'],n_res['AAC'])
		
		ci = evm.robust_concordance(n_res['Predicted'].values,n_res['AAC'].values)
		nm_res['Model'].append(model_name_map[nm])
		nm_res['Screen'].append("Pooled")
		nm_res['Concordance'].append(ci)
		nm_res['Pearson'].append(corr)

	bm_res = pd.DataFrame(bm_res)
	nm_res =pd.DataFrame(nm_res)
	ar = pd.concat([bm_res,nm_res])
	ar['Drug']=drug
	all_res.append(ar)

	
	for screen in ['gCSI','GDSC2','Pooled']:
		b= {}
		_df = bm_res[(bm_res['Screen']==screen)]
		for metric in ['Pearson','Concordance']:
			best_baseline = np.max(_df[metric])
			b[metric]= best_baseline

		n_sub = nm_res[nm_res['Screen']==screen]
		for idx, row in n_sub.iterrows():
			for metric in b:
				gaps['Drug'].append(drug)
				gaps['Metric'].append(metric)
				gaps['Model'].append(row['Model'])
				gaps['Screen'].append(screen)
				gaps['Gap'].append(row[metric]-b[metric])
				
	

all_res = pd.concat(all_res)
all_res.to_csv(f"../results/{src}_all.csv")
gaps = pd.DataFrame(gaps)
gaps.to_csv(f"../results/{src}_gaps.csv")

