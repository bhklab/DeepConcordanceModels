####
#
#
#	Script to process the PSet data into individual 
#	datasets and also plot distributions. 
#
######


import sys
import pandas as pd
import utils
import os
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import itertools as itl


def fetch_genes()->List[str]:
	
	genes = utils.read_text_file(f"../procdata/gene_sets/COSMIC.txt")+\
		utils.read_text_file(f"../procdata/gene_sets/KEGG.txt") +\
		utils.read_text_file("../procdata/gene_sets/ttd_targets.txt")
	
	genes = list(pd.unique(genes))
	return genes


screens = ['Broad','GDSC2','gCSI','GDSC1'] 
## Compute the common drugs across each one
ds_to_rx = {}
for screen in screens:
	resp = pd.read_csv(f"../procdata/{screen}/PSet_Data/responses.csv")
	ds_to_rx[screen] = list(pd.unique(resp['Drug']))


common_drugs = set(ds_to_rx['gCSI']).intersection(set(ds_to_rx['GDSC2']).intersection(set(ds_to_rx['Broad'])))
common_drugs = sorted(list(common_drugs))

with open("../procdata/drug_lists/common_drugs.txt","w") as ostream:
	ostream.writelines([x+"\n" for x in common_drugs])


genes = fetch_genes()

for screen in screens:
	
	# by construction these have all the same genes in the same order (protein coding)

	expr = pd.read_csv(f"../procdata/{screen}/PSet_Data/expression.csv")

	resp = pd.read_csv(f"../procdata/{screen}/PSet_Data/responses.csv")
	resp = resp[resp['Drug'].isin(common_drugs)]

	solid_ccls = utils.read_text_file(f"../procdata/{screen}/PSet_Data/solid_ccls.txt")
		

	expr = expr[expr['Cell Line'].isin(solid_ccls)]
	resp = resp[resp['Cell Line'].isin(solid_ccls)]
	

	_expr = expr[['Cell Line']+[gene for gene in genes if gene in expr.columns]]
	

	for drug in common_drugs:
		temp_resp = resp[resp['Drug']==drug][['Cell Line','AAC']]
		fig_path = f"../figs/{drug}/assay_wide/"
		os.makedirs(fig_path, exist_ok=True)
		sns.displot(data=temp_resp, x="AAC", fill=True,kind="kde")
		plt.title(f"{drug} AAC in {screen}")
		plt.tight_layout()
		plt.savefig(f"{fig_path}/{screen}_distribution.png",dpi=500)
		plt.close()


		opath = f"../procdata/{screen}/{drug}"
		os.makedirs(opath,exist_ok=True)
		data = _expr.merge(temp_resp, on='Cell Line')
		data.to_csv(f"{opath}/data.csv",index=False)

		_temp = temp_resp.sort_values(by=["AAC"],ascending=False)
		_temp['idx']= np.arange(_temp.shape[0])
		sns.scatterplot(data=_temp, x="idx",y="AAC")
		plt.title(f"Dose Response Profile for {drug} in {screen}")
		plt.savefig(f"{fig_path}/{screen}_profile.png",dpi=500)
		plt.close()



for drug in common_drugs:
	d1 = pd.read_csv(f"../procdata/GDSC2/{drug}/data.csv")
	d2 = pd.read_csv(f"../procdata/gCSI/{drug}/data.csv")
	d1['Cell Line'] = ["GDSC2_"+x for x in d1['Cell Line']]
	d2['Cell Line'] = ["gCSI_"+x for x in d2['Cell Line']]
	# d1 = d1[~d1['Cell Line'].isin(d2['Cell Line'])]

	data = pd.concat((d1,d2))
	opath = f"../procdata/GDSC2+gCSI/{drug}"
	os.makedirs(opath,exist_ok=True)
	data.to_csv(f"{opath}/data.csv",index=False)
