import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import os
import utils








####
# Plot Benchmarks

# ranking_benchmark_f = f"../results/{target_data}/{ccl_type}/ranking_metrics.csv"


benchmarks = pd.read_csv("../results/benchmarks/kfold/scalar_metrics_loss.csv")
# print(benchmarks.head())
# benchmarks = benchmarks[benchmarks['Eval. Type']!='All']

# drugs = pd.unique(benchmarks['Drug'])
drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")


# drugs = [x for x in drugs if x!='JQ1 compound']
# all_neur = pd.DataFrame()
# all_models = pd.DataFrame()
# neur_gaps = pd.DataFrame()
plot_delts = [0., 0.02, 0.05, 0.7, 0.1, 0.15, 0.2]
plot_eps = [5,10,15,30]
all_neurals = pd.DataFrame()
for drug in drugs:
	ecinet = pd.read_csv(f"../results/neurals/param_sweeps/{drug}/ECINET/scalar_metrics.csv")
	one_narrow = pd.read_csv(f"../results/neurals/param_sweeps/{drug}/ONE_NARROW/scalar_metrics.csv")
	one_wide = pd.read_csv(f"../results/neurals/param_sweeps/{drug}/ONE_WIDE/scalar_metrics.csv")
	two_narrow = pd.read_csv(f"../results/neurals/param_sweeps/{drug}/TWO_NARROW/scalar_metrics.csv")
	two_wide = pd.read_csv(f"../results/neurals/param_sweeps/{drug}/TWO_WIDE/scalar_metrics.csv")
	neurals = pd.concat((ecinet,one_narrow,one_wide,two_narrow,two_wide))
	
	# print([int(x.rstrip("_")[0])+1 for x in neurals['cv_score'].values])
	# print(list(pd.unique(neurals['cv_score'])))
	neurals['Epoch'] = [int(x.split("_")[0])+1 for x in neurals['cv_score'].values]
	# print(pd.unique(neurals['Tau_Q']))
	# sys.exit()
	neurals = neurals[neurals['Eval. Type']!='All']
	all_neurals = pd.concat((all_neurals,neurals))
	neurals = neurals[neurals['Delta'].isin(plot_delts)]
	neurals = neurals[neurals['Epoch'].isin(plot_eps)]


	for ds in ['gCSI','GDSC2']:
		for et in ['Seen','Unseen']:
			_temp = neurals[(neurals['Data Set']==ds) & (neurals['Eval. Type']==et)]
			_temp = _temp[['Drug','Data Set', 'Model','Eval. Type', 'Spearman Distance', 'Concordance','Pearson','Delta','Tau_Q','Epoch']]
			_temp.drop_duplicates(inplace=True)
			# print(_temp)
			
			for m in ['Concordance','Pearson']:
				ax = sns.catplot(_temp, x='Delta',y=m,hue='Model',errorbar=None,col='Epoch',kind='bar',row='Tau_Q')
				sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
				ax.figure.suptitle(f"{drug} {ds} {et}")
				ax.figure.subplots_adjust(top=.9)
				# plt.title(
				
				plt.legend(loc='upper left',bbox_to_anchor=(1, 1))
				plt.tight_layout()
				# plt.show()
				# sys.exit()
				figpath = f"../figs/{drug}/sweeps/{et}/{ds}"
				os.makedirs(figpath,exist_ok=True)
				plt.savefig(f"{figpath}/{m}_vs_delta_sweep.png")
				plt.close()
				# plt.show()
		
all_neurals.to_csv("all_neurals.csv")
print(all_neurals.head())
all_neurals = all_neurals[all_neurals['Epoch']==30]
for model in pd.unique(all_neurals['Model']):
	for ds in ['gCSI','GDSC2']:
		for et in ['Seen','Unseen']:
			_temp = all_neurals[(all_neurals['Data Set']==ds) & (all_neurals['Eval. Type']==et)]
			_temp = _temp[_temp['Model']==model]
			_temp = _temp[['Drug','Pearson','Concordance','Delta','Tau_Q']]
			_temp.drop_duplicates(inplace=True)
			sns.boxplot(_temp, x='Delta', y='Concordance',hue='Tau_Q')
			print(_temp)
			sys.exit()


			# for m in ['Concordance','Pearson']:



	