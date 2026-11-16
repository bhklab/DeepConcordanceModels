import utils 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import os


drug_list = utils.read_text_file("../procdata/drug_lists/post_qc.txt")
neural_models = ['ECINET','ONE_NARROW','TWO_NARROW']
eval_settings = ['Seen','Unseen']
eval_cols =  ['Drug','Data Set', 'Eval. Type', 'Model', 'Perm Overlap',	'NDCG','ndcg_binary','AP@K','P@K','R@K','Threshold','K','cv_score']
datasets = ['gCSI','GDSC2']
pearsons, cis = [],[]
benchmarks = pd.read_csv("../results/benchmarks/kfold/COSMIC+/ranking_metrics_Loss.csv",usecols=eval_cols)
benchmarks = benchmarks[benchmarks['Eval. Type'].isin(eval_settings)]
benchmarks = benchmarks[benchmarks['Model']!='Lasso']
plotting_ks = [5,10]

for drug in drug_list:
	scalars = [pd.read_csv(f"../results/neurals/kfold/COSMIC+/{drug}/{model}/ranking_metrics.csv",usecols=eval_cols) for model in neural_models]
	neurals = pd.concat(scalars,axis=0)
	neurals = neurals[neurals['Eval. Type'].isin(eval_settings)]
	neurals = neurals[neurals['cv_score']=='loss']

	all_res = pd.concat([neurals,benchmarks[benchmarks['Drug']==drug]])
	all_res = all_res[all_res['K'].isin(plotting_ks)]
	all_res.drop_duplicates(inplace=True)



	for et in ['Unseen']:
		for ds in datasets:
			temp = all_res[(all_res['Eval. Type']==et) & (all_res['Data Set']==ds)]
			for t in [20]:
				temp = temp[temp['Threshold']==t]
				for metric in ['P@K']:
					sns.barplot(temp,x='K',y=metric,hue='Model')
					plt.title(f"{drug} - {et} - {ds} - {t}\n{metric}")
					plt.tight_layout()
					plt.savefig(f"./ranking_figs/{drug}_{ds}.png")
					plt.close()
