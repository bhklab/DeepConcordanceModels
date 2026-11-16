import pandas as pd

SIZE_CUTOFF= 500
CORR_CUTOFF = 0.25



with open("../procdata/drug_lists/common_drugs.txt","r") as ostream:
	lines = ostream.readlines()
	common_drugs = [x.rstrip() for x in lines]


corrs = pd.read_csv("../results/correlations.csv",index_col=0)
sizes = pd.read_csv("../results/dataset_sizes.csv",usecols=["Drug","Train Screen","Num. Train"])
sizes = sizes[sizes['Train Screen']=='Broad']
sizes = sizes[sizes['Num. Train']>=SIZE_CUTOFF]
# print(sizes.columns)
size_drugs = set(sizes['Drug'])
filtering_list = set(common_drugs)

corrs = corrs[(corrs["First Screen"]=="Broad") & (corrs['Source']=='Assay')]
corrs = corrs[corrs['Pearson']>=CORR_CUTOFF]

for screen in ['gCSI','GDSC1','GDSC2']:
	_t = corrs[corrs['Second Screen']==screen]
	filtering_list = filtering_list.intersection(set(_t['Drug']))
# filtering_list = corrs['Drug']


filtering_list = filtering_list.intersection(size_drugs)
filtering_list=sorted(list(filtering_list))

with open("../procdata/drug_lists/post_qc.txt","w") as ostream:
	ostream.writelines([x+"\n" for x in filtering_list])
	