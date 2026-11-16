import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import os




####
# Plot Benchmarks

# ranking_benchmark_f = f"../results/{target_data}/{ccl_type}/ranking_metrics.csv"


benchmarks = pd.read_csv("../results/benchmarks/kfold/ranking_metrics_loss.csv")
print(benchmarks.head())
benchmarks = benchmarks[benchmarks['Eval. Type']!='All']
drugs = pd.unique(benchmarks['Drug'])

all_neur = pd.DataFrame()
all_models = pd.DataFrame()
neur_gaps = pd.DataFrame()
plotting_ks = [3,5,10,15,20]
all_data = pd.DataFrame()
for drug in drugs:
	ecinet = pd.read_csv(f"../results/neurals/kfold/{drug}/ECINET/ranking_metrics.csv")
	print(ecinet.head())
	sys.exit()
	one_narrow = pd.read_csv(f"../results/neurals/kfold/{drug}/ONE_NARROW/ranking_metrics.csv")
	one_wide = pd.read_csv(f"../results/neurals/kfold/{drug}/ONE_WIDE/ranking_metrics.csv")
	two_narrow = pd.read_csv(f"../results/neurals/kfold/{drug}/TWO_NARROW/ranking_metrics.csv")
	two_wide = pd.read_csv(f"../results/neurals/kfold/{drug}/TWO_WIDE/ranking_metrics.csv")
	neurals = pd.concat((ecinet,one_narrow,one_wide,two_narrow,two_wide))
	neurals = neurals[neurals['cv_score']=='ci']
	neurals = neurals[neurals['Eval. Type']!='All']

	drug_bench = benchmarks[benchmarks['Drug']==drug]
	
	data = pd.concat((neurals,drug_bench))
	data = data[data['K'].isin(plotting_ks)][['Drug','Data Set','Eval. Type', 'Model','Perm Overlap','K']]
	# for ds in ['GDSC2','gCSI']:
	# 	for et in ['Seen','Unseen']:
	# 		temp = data[(data['Data Set']==ds) & (data['Eval. Type']==et)]
	# 		temp = temp[temp['Model'].isin(['ElasticNet','RandomForest','ECINET','ONE_NARROW','TWO_NARROW'])]
			
	# 		sns.barplot(temp,x='K',y= 'Perm Overlap',hue='Model')
	# 		plt.title(f"{drug} Permutation Overlap\n {ds} {et}")
	# 		plt.show()

	all_data = pd.concat((all_data,data))

print(all_data.head())
all_data = all_data[['Data Set','Eval. Type', 'Model','Perm Overlap','K']]
df = all_data.groupby(['Data Set','Eval. Type', 'Model','K']).mean()
df.reset_index(inplace=True)
print(df)
df = df[df['Model'].isin(['ECINET','ONE_NARROW','ONE_WIDE','TWO_NARROW','TWO_WIDE'])]
for ds in ['GDSC2','gCSI']:
	for et in ['Seen','Unseen']:
		temp = df[(df['Data Set']==ds) & (df['Eval. Type']==et)]
		# print(temp)
		# sys.exit()
		sns.barplot(temp,x='K',y= 'Perm Overlap',hue='Model')
		plt.title(f" Permutation Overlap\n {ds} {et}")
		plt.show()
sys.exit()
	