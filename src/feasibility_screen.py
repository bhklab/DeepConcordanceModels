##
#
#
#
#	Pairwise Comparisons
#
#
###

from collections import defaultdict
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
import itertools as itl
import os

def summarize_response_set(drug,resp):
	res_dict = {}
	res_dict['Drug'] = drug
	res_dict['Max'] = max(resp).item()
	res_dict['Min'] = min(resp).item()
	res_dict['IQR'] = iqr(resp).item()
	res_dict['Mean'] = np.mean(resp).item()
	res_dict['Median'] = np.median(resp).item()
	return res_dict


base_path = "../procdata"
screens = ['Broad','gCSI','GDSC2','GDSC1']

with open("../procdata/drug_lists/common_drugs.txt", "r") as istream:
	drugs = istream.readlines()



drugs = [x.rstrip() for x in drugs]

corr_result = defaultdict(list)
sizes = defaultdict(list)


for screen_A, screen_B in itl.combinations(screens,2):
	# print(f"{screen_A} vs {screen_B}")
	# print(screen_B)`
	if screen_B == "GDSC2+gCSI" and screen_A in ['GDSC2','gCSI']:
		continue
	
	for drug in drugs:


		df1 =  pd.read_csv(f"{base_path}/{screen_A}/{drug}/data.csv",usecols = ["Cell Line","AAC"])
		df2 = pd.read_csv(f"{base_path}/{screen_B}/{drug}/data.csv",usecols = ["Cell Line","AAC"])
		
		if screen_B == "GDSC2+gCSI":
			df2['Cell Line'] = ["_".join(x.split("_")[1:]) for x in df2['Cell Line']]
			
		common = df1.merge(df2,on='Cell Line')
		common.columns =  ['Cell Line', screen_A,screen_B]
		

		if screen_A == 'Broad':
			sizes['Drug'].append(drug)
			sizes['Train Screen'].append("Broad")
			sizes['Test Screen'].append(screen_B)
			sizes['Num. Train'].append(df1.shape[0])
			sizes['Num. Test'].append(df2.shape[0])
			sizes['Num. Seen'].append(common.shape[0])
			sizes['Num. Unseen'].append(df2.shape[0]-common.shape[0])

		
		corr_table = common[[screen_A,screen_B]].corr()
		corr = np.round(corr_table.iloc[0,1],4)

		corr_result['Drug'].append(drug)
		corr_result['Source'].append('Dataset')
		corr_result['First Screen'].append(screen_A)
		corr_result['Second Screen'].append(screen_B)
		corr_result['Pearson'].append(corr)


		fig_path = f"../figs/{drug}/data_set"
		os.makedirs(fig_path,exist_ok=True)
		

		sns.scatterplot(data = common, x=screen_A, y=screen_B)
		plt.title(f"AAC Comparison - {drug}\n Correlation = {corr}")
		plt.plot(common[screen_A], common[screen_A], color='red')
		plt.axvline(x = 0.2)
		plt.axhline(y = 0.2)
		plt.savefig(f"{fig_path}/{screen_A}_vs_{screen_B}.png",dpi=500)
		plt.close()



		if screen_A != 'GDSC2+gCSI' and screen_B != 'GDSC2+gCSI':
			fig_path = f"../figs/{drug}/assay_wide"
			os.makedirs(fig_path,exist_ok=True)
			df1 =  pd.read_csv(f"{base_path}/{screen_A}/PSet_Data/responses.csv")
			df2 = pd.read_csv(f"{base_path}/{screen_B}//PSet_Data/responses.csv")
			df1 = df1[df1['Drug']==drug]
			df2 = df2[df2['Drug']==drug]
			df1 = df1[['Cell Line','AAC']]
			df2 = df2[['Cell Line','AAC']]

			common = df1.merge(df2,on='Cell Line')
			common.columns =  ['Cell Line', screen_A,screen_B]
			corr_table = common[[screen_A,screen_B]].corr()
			corr = np.round(corr_table.iloc[0,1],4)
			corr_result['Drug'].append(drug)
			corr_result['Source'].append('Assay')
			corr_result['First Screen'].append(screen_A)
			corr_result['Second Screen'].append(screen_B)
			corr_result['Pearson'].append(corr)

			sns.scatterplot(data = common, x=screen_A, y=screen_B)
			plt.title(f"AAC Comparison - {drug}\n Correlation = {corr}")
			plt.plot(common[screen_A], common[screen_A], color='red')
			plt.axvline(x = 0.2)
			plt.axhline(y = 0.2)
			plt.savefig(f"{fig_path}/{screen_A}_vs_{screen_B}.png",dpi=500)
			plt.close()






sizes = pd.DataFrame(sizes)
corr = pd.DataFrame(corr_result)


sizes.to_csv("../results/dataset_sizes.csv")
corr.to_csv("../results/correlations.csv")

for screen in screens:
	for drug in drugs:
		df =  pd.read_csv(f"{base_path}/{screen}/{drug}/data.csv",usecols = ["Cell Line","AAC"])
		fig_path = f"../figs/{drug}/data_set"

		sns.displot(data=df, x="AAC", fill=True,kind="kde")
		plt.title(f"{drug} AAC in {screen}")
		plt.tight_layout()
		plt.savefig(f"{fig_path}/{screen}_distribution.png",dpi=500)
		plt.close()

		_temp = df.sort_values(by=["AAC"],ascending=False)
		_temp['idx']= np.arange(_temp.shape[0])
		sns.scatterplot(data=_temp, x="idx",y="AAC")
		plt.title(f"Dose Response Profile for {drug} in {screen}")
		plt.savefig(f"{fig_path}/{screen}_profile.png",dpi=500)
		plt.close()
