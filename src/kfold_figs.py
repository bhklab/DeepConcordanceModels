import utils 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import os


drug_list = utils.read_text_file("../procdata/drug_lists/post_qc.txt")
neural_models = ['ECINET','One Layer','Two Layer']
eval_settings = ['Seen','Unseen']
eval_cols =  ['Drug','Data Set', 'Eval. Type', 'Model', 'Pearson','Concordance', 'Delta',  'cv_score']
datasets = ['gCSI','GDSC2']
pearsons, cis = [],[]
benchmarks = pd.read_csv("../results/benchmarks/kfold/COSMIC+/scalar_metrics_Loss.csv",usecols=eval_cols)
benchmarks = benchmarks[benchmarks['Eval. Type'].isin(eval_settings)]
benchmarks = benchmarks[benchmarks['Model']!='Lasso']
neur_gaps = pd.DataFrame()


for drug in drug_list:
	scalars = [pd.read_csv for model in neural_models]

	ecinet = pd.read_csv(f"../results/neurals/kfold/COSMIC+/{drug}/ECINET/scalar_metrics.csv",usecols=eval_cols)
	
	one_narrow = pd.read_csv(f"../results/neurals/kfold/COSMIC+/{drug}/ONE_NARROW/scalar_metrics.csv",usecols=eval_cols)
	two_narrow = pd.read_csv(f"../results/neurals/kfold/COSMIC+/{drug}/TWO_NARROW/scalar_metrics.csv",usecols=eval_cols)
	one_narrow['Model'] = "One Layer"
	two_narrow['Model'] = "Two Layer"
	neurals = pd.concat([ecinet,one_narrow,two_narrow],axis=0)
	neurals = neurals[neurals['Eval. Type'].isin(eval_settings)]

	neurals = neurals[neurals['cv_score']=='loss']
	all_res = pd.concat([neurals,benchmarks[benchmarks['Drug']==drug]])
	all_res.drop_duplicates(inplace=True)
	
	for et in eval_settings:
		for ds in datasets:
			temp = all_res[(all_res['Eval. Type']==et) & (all_res['Data Set']==ds)]
			print(temp)
			n_temp =temp[temp['Model'].isin(neural_models)]
			b_temp = temp[~temp['Model'].isin(neural_models)]
			
			ng = n_temp.copy()
			ng['Concordance'] = ng['Concordance']-np.max(b_temp['Concordance'])
			ng['Pearson'] = ng['Pearson'] - np.max(b_temp['Pearson'])
			ng['Sig. Conc'] = np.max(b_temp['Concordance'])>=0.6
			

			neur_gaps = pd.concat((neur_gaps,ng))
	

print("scump")

for ds in ['GDSC2','gCSI']:
	for et in ['Seen','Unseen']:
		temp = neur_gaps[(neur_gaps['Data Set']==ds) & (neur_gaps['Eval. Type']==et)]
		print(temp)
		for metric in ['Pearson','Concordance']:
			os.makedirs(f"./scalar_figs/{metric}/",exist_ok=True)
			
			sns.barplot(temp, y='Drug', x=metric,hue='Model')
			plt.title(f"Perfomance Relative to Best Baseline\n{et} Cell Lines in {ds}")
			plt.xlabel(f"{metric} performance gap")
			plt.tight_layout()
			
			plt.savefig(f"./scalar_figs/{metric}/{ds}_{et}_gaps.png")
			plt.close()

			os.makedirs(f"./scalar_figs/{metric}/signif/",exist_ok=True)
			_temp = temp[temp['Sig. Conc']==True]
			sns.barplot(_temp, y='Drug', x=metric,hue='Model')
			plt.title(f"Perfomance Relative to Best Baseline\n{et} Cell Lines in {ds}")
			plt.xlabel(f"{metric} performance gap")
			plt.tight_layout()
			
			plt.savefig(f"./scalar_figs/{metric}/signif/{ds}_{et}_gaps_signif.png")
			plt.close()
			

# print(pd.value_counts(pearsons))
