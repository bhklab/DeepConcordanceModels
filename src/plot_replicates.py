###
#
#
# 
#
####

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import os
import utils
import evaluation_metrics as evm
from scipy.stats import pearsonr, mannwhitneyu, ttest_ind
from collections import defaultdict


neural_to_model = {'ECINET':"ECINET",'ONE_NARROW':"One Layer", "TWO_NARROW":"Two Layer"}

baselines, neurals = [],[]

gaps = pd.read_csv("../results/Broad/empirical_deltas.csv",index_col=0)
gaps = gaps[gaps['Type']=='All']


deltas = {k:v for k,v in zip([0]+list(gaps['Percentile']),[0]+list(gaps['Delta']))}

### Key question: does doing delta work
screens = ['gCSI','GDSC2']
# screens = ['Broad']
drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")
# drugs = drugs[:5]

neural_results = defaultdict(list)
ep = 30

for screen in screens:
	for drug in drugs:
		for model in neural_to_model:
			neural = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/test/{screen}/{model}.csv",index_col=0)
			
			neural = neural[neural['Epoch'].isin([ep])]
			for p in deltas:
				_temp = neural[(neural['Percentile']==p) & (neural['Epoch']==ep)]
				corr,_ = pearsonr(_temp['Predicted'],_temp['AAC'])
				ci = evm.robust_concordance(_temp['Predicted'],_temp['AAC'])
				neural_results['Model'].append(neural_to_model[model])
				neural_results['Drug'].append(drug)
				neural_results['Screen'].append(screen)
				neural_results['Percentile'].append(p)
				neural_results['Delta'].append(deltas[p])
				# neural_results['Epoch'].append(e)
				neural_results['Pearson'].append(corr)
				neural_results['Concordance'].append(ci)

neural_results = pd.DataFrame(neural_results)
neural_results.to_csv("../results/rg_deltas.csv")
# neural_results['Binned Delta'] = ["Positive" if x>0 else "Zero" for x in neural_results['Delta'].values]

# comps = defaultdict(list)
# for screen in screens:
# 	temp = neural_results[neural_results['Screen']==screen]
# 	for metric in ['Pearson','Concordance']:
# 		sns.boxplot(temp,x = "Binned Delta",y=metric,hue = "Binned Delta",palette="tab10",boxprops=dict(alpha=.3))
# 		sns.stripplot(temp, x="Binned Delta", y=metric, size=4, color=".3")
# 		plt.legend([],[], frameon=False)
# 		plt.savefig(f"../figs/{screen}_repgaps_binned_deltas_{metric}.png",dpi=500)
# 		plt.close()
		
# 		for pct in [75,85,95]:
# 			zed = temp[temp['Percentile']==0][metric].values
# 			pos = temp[temp['Percentile']==pct][metric].values
# 			comps['Screen'].append(screen)
# 			comps['Metric'].append(metric)
# 			comps['Delta'].append(pct)
# 			_,p = ttest_ind(zed,pos,equal_var = False,alternative = 'less')
# 			comps['ttest p'].append(p)
# 			_,p = mannwhitneyu(zed,pos)
# 			comps['MW p'].append(p)


# 		sns.boxplot(temp,x = "Percentile",y=metric,hue = "Percentile",palette="tab10",boxprops=dict(alpha=.3))
# 		sns.stripplot(temp, x="Percentile", y=metric, size=4, color=".3")
# 		plt.legend([],[], frameon=False)
# 		plt.savefig(f"../figs/{screen}_repgaps_deltas_{metric}.png",dpi=500)
# 		plt.close()

# comps = pd.DataFrame(comps)
# comps.to_csv("../results/rg_pvals.csv")
# # for metric in ['Pearson','Concordance']:
# # 	sns.boxplot(neural_results,x = "Binned Delta",y=metric,hue = "Binned Delta",palette="tab10",boxprops=dict(alpha=.3))
# # 	sns.stripplot(neural_results, x="Binned Delta", y=metric, size=4, color=".3")
# # 	plt.legend([],[], frameon=False)
# # 	plt.savefig(f"../figs/pooled_repgaps_binned_deltas_{metric}.png",dpi=500)
# # 	plt.close()


# # 	pos = temp[temp['Binned Delta']=='Positive'][metric].values
# # 	zed = temp[temp['Binned Delta']=='Zero'][metric].values

	
	
# # 	comps['Screen'].append(screen)
# # 	comps['Metric'].append(metric)
# # 	_,p = ttest_ind(zed,pos,equal_var = False,alternative = 'less')
# # 	comps['ttest p'].append(p)
# # 	_,p = mannwhitneyu(zed,pos)
# # 	comps['MW p'].append(p)


# # 	sns.boxplot(neural_results,x = "Percentile",y=metric,hue = "Percentile",palette="tab10",boxprops=dict(alpha=.3))
# # 	sns.stripplot(neural_results, x="Percentile", y=metric, size=4, color=".3")
# # 	plt.legend([],[], frameon=False)
# # 	plt.savefig(f"../figs/pooled_repgaps_deltas_{metric}.png",dpi=500)
# # 	plt.close()
# # # print(neural_results)
# # # neural_results.to_csv("../results/rg_neurals.csv")



# # results = defaultdict(list)
# # for screen in screens:
# # 	for model in pd.unique(neural_results['Model']):
# # 		for drug in pd.unique(neural_results['Drug']):
# # 			_temp = neural_results[ (neural_results['Model']==model) &
# # 									(neural_results['Drug']==drug) &
# # 									(neural_results['Screen']==screen)]

# # 			for metric in ['Concordance', 'Pearson']:
# # 				best_val = np.max(_temp[metric])
# # 			# best_corr = np.max(_temp['Pearson'])
# # 				best_delta = _temp[_temp[metric]==best_val]['Percentile'].values[0]				
# # 				results['Drug'].append(drug)
# # 				results['Model'].append(model)
# # 				results['Metric'].append(metric)
# # 				results['Screen'].append(screen)
# # 				results['Delta Pct.'].append(best_delta)
# # results = pd.DataFrame(results)
# # for screen in screens:				
# # 	for metric in ['Concordance', 'Pearson']:
# # 		_t = results[(results['Metric']==metric) & (results['Screen']==screen)]
# # 		print(f"{screen} - {metric}")
# # 		print(_t['Delta Pct.'].value_counts())
