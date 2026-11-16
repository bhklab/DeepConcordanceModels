####
##
#
#	Code to analyze the preprocessed cell line screen data
# 	and visualize both data set statistics as well as
#	drug response information.
#
#	Produces:
#		- Venn Diagrams of Data Set Sizes
#		- Distributions of AAC Values
#		- Comparisons of AAC Values in the same CCL and drug
#			in different screens.
#		- A table of data set sizes in various evaluation scenarios.
#
#		
#	
#	Author: James Bannon
#	Github: @jbannon
#
#
#
#	Note: this code is a mess and inefficient; it exists solely to understand the data available. 
#
#	
#####

import sys
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np  
import seaborn as sns
import utils 
from matplotlib_venn import venn3
import itertools as itl
from collections import defaultdict
import os


# set up dictionaries mapping data sets to different cell line types
ds_to_ccl, ds_to_rx,ds_to_solid, ds_to_liquid = {},{},{},{}
ds_to_ccl_e, ds_to_solid_e,ds_to_liquid_e = {},{},{}

base_path = "../procdata"
ds_names = ['gCSI','GDSC2','Broad','GDSC1']

ds_counts = defaultdict(list)


# below we iterate over each dataset and load in the c
for dataset in ds_names:
	response = pd.read_csv(f"{base_path}/{dataset}/PSet_Data/responses.csv")
	ccl_data = pd.read_csv(f"{base_path}/{dataset}/PSet_Data/ccl_data.csv")
	expression = pd.read_csv(f"{base_path}/{dataset}/PSet_Data/expression.csv",usecols=[0,1])
	
	solid_ccls = set(ccl_data[~ccl_data['tissueid'].isin(['Lymphoid','Myeloid'])]['Cellosaurus.Accession.id'].values)
	
	liquid_ccls = set(ccl_data[ccl_data['tissueid'].isin(['Lymphoid','Myeloid'])]['Cellosaurus.Accession.id'].values)
	
	ds_to_ccl[dataset]= [x for x in pd.unique(ccl_data['Cellosaurus.Accession.id'].values)]
	ds_to_rx[dataset] = [x for x in response['Drug'].values]
	ds_to_solid[dataset] = solid_ccls
	ds_to_liquid[dataset] = liquid_ccls

	ds_to_ccl_e[dataset] = [x for x in pd.unique(expression['Cell Line'].values)]
	ds_to_solid_e[dataset] = [x for x in pd.unique(expression['Cell Line'].values) if x in solid_ccls]

	ds_to_liquid_e[dataset] = [x for x in pd.unique(expression['Cell Line'].values) if x in liquid_ccls]


	for data_type, df, col in zip(
		['response','ccl','expression'],
		[response,ccl_data,expression],
		['Cell Line', 'Cellosaurus.Accession.id','Cell Line']):

		num_ccls = len(pd.unique(df[col]))
		num_solid = len(pd.unique(df[df[col].isin(solid_ccls)][col]))
		num_liquid = len(pd.unique(df[df[col].isin(liquid_ccls)][col]))
		# 	ostream.writelines([x+"\n" for x in ds_to_ccl[dataset]])		
		ds_counts['Screen'].append(dataset)
		ds_counts['Data Type'].append(data_type)
		ds_counts['Total CCLs'].append(num_ccls)
		ds_counts['Num. Liquid'].append(num_liquid)
		ds_counts['Num. Solid'].append(num_solid)
		ds_counts['Check'].append(num_solid + num_liquid)


CCL_df = pd.DataFrame(ds_counts)
CCL_df.to_csv("../results/dataset_ccl_counts.csv")



venn3([set(ds_to_rx['gCSI']),set(ds_to_rx['GDSC2']),set(ds_to_rx['Broad'])],['gCSI','GDSC2','Broad'])
plt.title("Venn Diagram of Tested Compounds")
plt.savefig("../figs/Compound_Venn.tif",dpi=300)
plt.close()

## drugs commmon to all testing
drug_intersection = set(ds_to_rx['gCSI']).intersection(set(ds_to_rx['GDSC2']).intersection(set(ds_to_rx['Broad'])))
drug_intersection = sorted(list(drug_intersection))

with open("../procdata/drug_lists/common_drugs.txt","w") as ostream:
	ostream.writelines([x+"\n" for x in drug_intersection])



venn3([set(ds_to_ccl['gCSI']),set(ds_to_ccl['GDSC2']),set(ds_to_ccl['Broad'])],['gCSI','GDSC2','Broad'])
plt.title("Venn Diagram of Screened CCLs")
plt.savefig("../figs/CCL_Venn_meta.png")
plt.close()


venn3([set(ds_to_solid['gCSI']),set(ds_to_solid['GDSC2']),set(ds_to_solid['Broad'])],['gCSI','GDSC2','Broad'])
plt.title("Venn Diagram of Screened Solid Tumor CCLs")
plt.savefig("../figs/Solid_CCL_Venn.png")
plt.close()

venn3([set(ds_to_liquid['gCSI']),set(ds_to_liquid['GDSC2']),set(ds_to_liquid['Broad'])],['gCSI','GDSC2','Broad'])
plt.title("Venn Diagram of Screened Liquid Tumor CCLs")
plt.savefig("../figs/Liquid_CCL_Venn_meta.png")
plt.close()



## now plot expression 
venn3([set(ds_to_ccl_e['gCSI']),set(ds_to_ccl_e['GDSC2']),set(ds_to_ccl_e['Broad'])],['gCSI','GDSC2','Broad'])
plt.title("Venn Diagram of Screened CCLs")
plt.savefig("../figs/CCL_Venn_expr.png")
plt.close()


venn3([set(ds_to_solid_e['gCSI']),set(ds_to_solid_e['GDSC2']),set(ds_to_solid_e['Broad'])],['gCSI','GDSC2','Broad'])
plt.title("Venn Diagram of Screened Solid Tumor CCLs")
plt.savefig("../figs/Solid_CCL_Venn_expr.png")
plt.close()

venn3([set(ds_to_liquid_e['gCSI']),set(ds_to_liquid_e['GDSC2']),set(ds_to_liquid_e['Broad'])],['gCSI','GDSC2','Broad'])
plt.title("Venn Diagram of Screened Liquid Tumor CCLs")
plt.savefig("../figs/Liquid_CCL_Venn_expr.png")
plt.close()



### Dose Response Curves
for drug in drug_intersection:
	
	all_subdir = f"../figs/{drug}/All/"
	solid_subdir = f"../figs/{drug}/Solid/"
	
	for ds_A,ds_B in itl.combinations(ds_names,2):

		response_A = pd.read_csv(f"{base_path}/{ds_A}/responses.csv")
		response_B = pd.read_csv(f"{base_path}/{ds_B}/responses.csv")
		# expr_A  =  pd.read_csv(f"{base_path}/{ds_A}/expression_lite.csv",usecols=[0,1])
		# expr_B = pd.read_csv(f"{base_path}/{ds_B}/expression_lite.csv",usecols = [0,1])


		for ccl_type, fig_subdir in zip(['All','Solid'],[all_subdir,solid_subdir]):
			os.makedirs(fig_subdir,exist_ok=True)
			temp_A = response_A[response_A['Drug']==drug][['Cell Line','AAC']]

			if ccl_type == 'Solid':
				temp_A = temp_A[temp_A['Cell Line'].isin(ds_to_solid[ds_A])]
			sns.displot(data=temp_A, x="AAC", fill=True,kind="kde")
			plt.title(f"{drug} AAC in {ds_A}")
			plt.tight_layout()
			plt.savefig(f"{fig_subdir}{ds_A}_distribution.jpeg")
			plt.close()

			_A = temp_A.sort_values(by=["AAC"],ascending=False)
			_A['idx']= np.arange(_A.shape[0])
			sns.scatterplot(data=_A, x="idx",y="AAC")
			plt.title(f"Dose Response Profile for {drug} in {ds_A}")
			plt.savefig(f"{fig_subdir}{ds_A}_profile.jpeg")
			plt.close()
			

			temp_B = response_B[response_B['Drug'] ==drug][['Cell Line','AAC']]

			if ccl_type == 'Solid':
				temp_B = temp_B[temp_B['Cell Line'].isin(ds_to_solid[ds_B])]
			all_resp = temp_A.merge(temp_B,on='Cell Line')
			all_resp.columns = ['Cell Line', f"{ds_A}",f"{ds_B}"]
			corr_table = all_resp[[f"{ds_A}",f"{ds_B}"]].corr()
			corr = np.round(corr_table.iloc[0,1],4)
			sns.displot(data=temp_B, x="AAC", fill=True,kind="kde")
			plt.title(f"{drug} AAC in {ds_B}")
			plt.tight_layout()
			plt.savefig(f"{fig_subdir}{ds_B}_distribution.jpeg")
			plt.close()

			_B = temp_B.sort_values(by=["AAC"],ascending=False)
			_B['idx']= np.arange(_B.shape[0])
			sns.scatterplot(data=_B, x="idx",y="AAC")
			plt.title(f"Dose Response Profile for {drug} in {ds_B}")
			plt.savefig(f"{fig_subdir}{ds_B}_profile.jpeg")
			plt.close()
			

			sns.scatterplot(data = all_resp, x=ds_A, y=ds_B)
			plt.title(f"AAC Comparison - {drug}\n Correlation = {corr}")
			plt.plot(all_resp[ds_A], all_resp[ds_A], color='red')
			plt.axvline(x=0.2)
			plt.axhline(y=0.2)
			plt.savefig(f"{fig_subdir}{ds_A}_vs_{ds_B}.jpeg")
			plt.close()


		
	


drug_to_responses = defaultdict(list)
drug_to_solid_responses = defaultdict(list)

for _ds in ds_names:
	response = pd.read_csv(f"{base_path}/{_ds}/responses.csv")

	for drug in drug_intersection:
		_temp = response[response['Drug']==drug]
		drug_to_responses[drug].extend(_temp['AAC'].values)
		_temp = _temp[_temp['Cell Line'].isin(ds_to_solid[_ds])]
		drug_to_solid_responses[drug].extend(_temp['AAC'].values)


for drug in drug_intersection:
	fig_subdir = f"../figs/{drug}/"
	drug_df = pd.DataFrame({'AAC':drug_to_responses[drug]})

	sns.displot(data=drug_df, x="AAC", fill=True,kind="kde")
	plt.title(f"{drug} AAC in All Screens")
	plt.tight_layout()
	plt.savefig(f"{fig_subdir}all_screens_distribution.jpeg")
	plt.close()

	drug_df	= drug_df.sort_values(by=["AAC"],ascending=False)
	drug_df['idx']= np.arange(drug_df.shape[0])

	sns.scatterplot(data=drug_df, x="idx",y="AAC")
	plt.title(f"{drug} Response Profile in All Screens")
	plt.tight_layout()
	plt.savefig(f"{fig_subdir}all_screens_profile.jpeg")
	plt.close()




