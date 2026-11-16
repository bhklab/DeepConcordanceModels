import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import os
import utils


all_neurals = pd.read_csv("all_neurals.csv")

all_neurals = all_neurals[all_neurals['Model'].isin(['ONE_NARROW','ONE_WIDE','TWO_NARROW','TWO_WIDE'])]

all_neurals = all_neurals[all_neurals['Epoch']==25]
print(all_neurals.head())
for drug in pd.unique(all_neurals['Drug']):
	for ds in ['gCSI','GDSC2']:
		for et in ['Seen','Unseen']:
			_temp = all_neurals[(all_neurals['Data Set']==ds) & (all_neurals['Eval. Type']==et) & (all_neurals['Drug']==drug)]
			_t = _temp[['Model','Pearson','Concordance']]
			_t.drop_duplicates(inplace=True)
			print(_t)
			sys.exit()
	# te
sys.exit()


for model in pd.unique(all_neurals['Model']):
	for ds in ['gCSI','GDSC2']:
		for et in ['Seen','Unseen']:
			_temp = all_neurals[(all_neurals['Data Set']==ds) & (all_neurals['Eval. Type']==et)]
			_temp = _temp[_temp['Model']==model]
			_temp = _temp[['Drug','Pearson','Concordance','Delta','Tau_Q']]
			_temp.drop_duplicates(inplace=True)

			for m in ['Concordance','Pearson']:
				sns.boxplot(_temp, x='Delta', y=m,hue='Tau_Q')
				plt.title(f"{model} - all drugs\n {ds} {et}")
				plt.savefig(f"../figs/{model}_{m}_{et}_{ds}_sweep.png")
				plt.close()
			# print(_temp)

