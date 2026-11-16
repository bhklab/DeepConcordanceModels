import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import os



ranking_metrics = pd.read_csv("../results/neurals/HIGH_TAU/ranking_metrics.csv")

scalar_metrics = pd.read_csv("../results/neurals/HIGH_TAU/scalar_metrics.csv")


print(ranking_metrics.head())
print(pd.unique(ranking_metrics['Delta']))



# print(ranking_metrics.columns)





plotting_ks = [1,5,10,15]
plotting_taus = ["50%","70%","90%", "95%"]
ranking_metrics=ranking_metrics[ranking_metrics['Drug'].isin(['Erlotinib','Crizotinib'])]
# prnt(ranking_metrics.head())
for metric in ['Perm Overlap', 'NDCG', 'Ordinal NDCG']:
	od = f"./tempfigs/{metric}/"
	# mkdir()
	os.makedirs(od,exist_ok=True)
	for ds in pd.unique(ranking_metrics['Data Set']):
			for delta in pd.unique(ranking_metrics['Delta']):
				temp = ranking_metrics[(ranking_metrics['Delta']==delta)]
				temp = temp[temp['K'].isin(plotting_ks)]
				temp = temp[temp['Tau_Q'].isin(plotting_taus)]
				temp = temp[temp['Eval. Type'].isin(['Seen', 'Unseen'])]
				temp = temp[temp['Data Set']==ds]

				temp = temp[['K',metric, 'Eval. Type', 'Tau_Q']]
				temp = pd.DataFrame(temp.groupby(['K', 'Eval. Type', 'Tau_Q']).mean())
				
				temp.reset_index(inplace=True)

					# print(temp)
				# sys.exit()
				# print(f"Delta: - {delta}, Tau - {tau}")
				p = sns.catplot(temp,kind = 'bar',x='K',y=metric, row = 'Eval. Type', col = "Tau_Q")
				p.fig.suptitle(f"Delta = {delta}, Eval Data = {ds}")
				# plt.title(f"{drug} - {tau} - {delta}")
				plt.tight_layout()
				plt.savefig(f"{od}/{ds}_{delta}_all_avg.png")
				plt.close()
				# plt.tight_layout()
				# plt.show()


				temp = ranking_metrics[(ranking_metrics['Delta']==delta)]
				temp = temp[temp['K'].isin(plotting_ks)]
				temp = temp[temp['Tau_Q'].isin(plotting_taus)]
				temp = temp[temp['Eval. Type'].isin(['Seen', 'Unseen'])]
				temp = temp[temp['Data Set']==ds]

				temp = temp[['K',metric, 'Model', 'Eval. Type', 'Tau_Q']]
				temp = pd.DataFrame(temp.groupby(['K', 'Model', 'Eval. Type', 'Tau_Q']).mean())
				temp.reset_index(inplace=True)

					# print(temp)
				# sys.exit()
				# print(f"Delta: - {delta}, Tau - {tau}")
				p = sns.catplot(temp,kind = 'bar',x='K',y=metric, hue = 'Model', row = 'Eval. Type', col = "Tau_Q")
				p.fig.suptitle(f"Delta = {delta}, Eval Data = {ds}")
				plt.tight_layout()
				# plt.title(f"{drug} - {tau} - {delta}")
				plt.savefig(f"{od}/{ds}_{delta}_models.png")
				# plt.tight_layout()
				# plt.show()
				plt.close()
				# sys.exit()

		
			
