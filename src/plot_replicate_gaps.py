###
#
#
#	For each drug we have three options:
#		 - Common Drugs
#		 - Solid Cell Line Delta Values
#		 - All Assay Cell Line Delta Values
#
###

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import utils
import numpy as np 
from collections import defaultdict
import os
import sys

common_drugs = utils.read_text_file("../procdata/drug_lists/common_drugs.txt")
cut_points = [50,95]

dec = 5
k = 0
gap_stats = defaultdict(list)
for screen, fname in zip(
	['Broad','gCSI','GDSC2','GDSC1'], 
	['CTRPv2_replicates.csv',"gCSI_replicates.csv","GDSC2_replicates.csv",'GDSC1_replicates.csv']
	):
	
	replicates = pd.read_csv(f"../results/qc/{fname}",index_col=0)
	
	
	
	
	info = pd.read_csv(f"../procdata/{screen}/PSet_Data/ccl_data.csv")
	info = info[~info['tissueid'].isin(['Myeloid','Lymphoid'])]

	
	replicates = replicates.dropna(subset="Delta")
	replicates['Raw Delta'] = replicates['X.Coord']- replicates['Y.Coord']	
	

	rep_solid = replicates[replicates['CCL'].isin(info['sampleid'])]
	rep_solid = rep_solid.dropna(subset=['Delta'])
	gap_stats['Screen'].append(screen)
	gap_stats['Mean Delta'].append(np.round(np.mean(rep_solid['Raw Delta']),dec))
	gap_stats['Std Delta'].append(np.round(np.std(rep_solid['Raw Delta']),dec))
	gap_stats['10% Delta'].append(np.round(np.quantile(rep_solid['Raw Delta'],.10),dec))
	gap_stats['50% Delta'].append(np.round(np.quantile(rep_solid['Raw Delta'],.50),dec))
	gap_stats['95% Delta'].append(np.round(np.quantile(rep_solid['Raw Delta'],.95),dec))

	gap_stats['Mean Abs Delta'].append(np.round(np.mean(rep_solid['Delta']),dec))
	gap_stats['Std. Abs Delta'].append(np.round(np.std(rep_solid['Delta']),dec))
	gap_stats['10% Abs. Delta'].append(np.round(np.quantile(rep_solid['Delta'],.10),dec))
	gap_stats['50%  Abs. Delta'].append(np.round(np.quantile(rep_solid['Delta'],.50),dec))
	gap_stats['95% Abs. Delta'].append(np.round(np.quantile(rep_solid['Delta'],.95),dec))
	
	q = np.quantile(rep_solid['Delta'],95/100)
	rep_solid['Binned Percentile'] = [">95" if x>q else "<=95" for x in rep_solid['Delta'].values]
	

	rep_common = replicates[replicates['Drug'].isin(common_drugs)]


	plt.scatter(rep_common['X.Coord'],rep_common['Y.Coord'])
	plt.title(f"{screen} Screen Replicate Noise in Common Drugs")
	plt.xlabel("Replicate 1")
	plt.ylabel("Replicate 2")
	plt.savefig(f"../figs/{screen}/common_drug_replicate_noise.png",dpi=500)
	plt.close()
	
	color_map = plt.cm.Spectral_r
	fig=plt.figure(figsize=(10,9))
	ax = fig.add_subplot(111)
	image = plt.hexbin(rep_solid['X.Coord'].values,rep_solid['Y.Coord'].values,cmap=color_map,gridsize=20,mincnt=1,bins='log')
	# g = sns.jointplot(rep_solid, x = 'X.Coord',y='Y.Coord',kind='hist',bins=40, hue = "Binned Percentile",legend=False)
	# # sns.jointplot(rep_solid, x = 'X.Coord',y='Y.Coord',kind='hist',bins = 40,co')
	# plt.axline(xy1=(0,0.18),slope=1,color ='red', linestyle='--')
	# plt.axline(xy1=(0.18,0),slope=1,color ='red', linestyle='--')
	# # g.ax_marg_x.remove()
	# # g.ax_marg_y.remove()
	# # plt.title(f"{screen} Screen Replicate Noise in Solid CCLs")
	# plt.xlabel("Replicate 1")
	# plt.ylabel("Replicate 2")
	
	
	cb = plt.colorbar(image,spacing='uniform',extend='max')
	plt.xlabel("Replicate 1",fontsize=14)
	plt.ylabel("Replicate 2",fontsize=14)
	_s = screen if screen != 'Broad' else "CTRPv2"
	plt.title(f"Replicate Pairs in Solid Cell Lines in {_s}",fontsize=14)
	plt.savefig(f"../figs/{screen}/solid_replicate_noise.png",dpi=500)
	plt.close()
	
	


	plt.scatter(replicates['X.Coord'],replicates['Y.Coord'])
	plt.title(f"{screen} Screen Replicate Noise")
	plt.xlabel("Replicate 1")
	plt.ylabel("Replicate 2")
	plt.savefig(f"../figs/{screen}/replicate_noise.png",dpi=500)
	plt.close()


	deltas = defaultdict(list)
	

	for type, df in zip(
		['Solid'], 
		[rep_solid]
		):
		print(f"{type}\n")
		qs = []

		for cp in [10,50,75,80,85,95]:
			print(f"{cp}\n")
			cc = 100 - np.round((100-cp)/2,2)
			u = np.quantile(df['Raw Delta'],cc/100)
			l = np.quantile(df['Raw Delta'],(100-cc)/100)
			q = np.quantile(rep_solid['Delta'],cp/100)
			deltas['Type'].append(type)
			deltas['Percentile'].append(cp)
			deltas['Delta'].append(np.round(q,4))
			qs.append( (cp, u,l ))
		fix, ax = plt.subplots()
		ax.hist(df['Raw Delta'],color = 'grey',bins=100)
		ymax = ax.get_ylim()[1]
		for q, ls in zip( [x for x in qs if x[0] in cut_points] ,["--",'-.',"-"]): 
			line = plt.axvline(q[1],color = 'red', linestyle = ls,label = f"{q[0]}%")
			ax.legend([line],f"cp%")
			line2 = plt.axvline(q[2],color = 'red', linestyle = ls)

	# 		if q[0] == 95:
	# 			ax.text(q[1],0.5*ymax,q[1])
			# 
		_s = screen if screen != 'Broad' else "CTRPv2"
		plt.title(f"Histogram of AAC Gaps in {_s}")
		plt.xlabel("AAC Gap")
		plt.ylabel("Count")
		plt.legend()
		plt.savefig(f"../figs/{screen}/{type}_histogram.png",dpi=500)
		# plt.show()
		plt.close()
		print(screen)
		print(np.mean(rep_solid['Delta']))
		# sys.exit()
		
	
	deltas = pd.DataFrame(deltas)
	opath = f"../results/{screen}"
	# os.makedirs(opath,exist_ok=True)

	deltas.to_csv(f"{opath}/empirical_deltas.csv")
gap_stats = pd.DataFrame(gap_stats)
gap_stats.to_csv("../results/gap_stats.csv")




delta = 0.05915102013903200
drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")
df = pd.read_csv("../results/qc/CTRPv2_replicates.csv",index_col=0)
df = df[df['Drug'].isin(drugs)]



df = df[df['CCL'].isin(["A-549","CAL-51"])]



drug_to_list = {}
drug_to_delta = {}
res = defaultdict(list)
for drug in pd.unique(df['Drug']):
    for ccl in pd.unique(df['CCL']):
        _t = df[(df['Drug']==drug) & (df['CCL']==ccl)]
        reps = set(_t['X.Coord']).union(set(_t['Y.Coord']))
        res['AAC'].extend(sorted(list(reps)))
        res['Drug'].extend(len(reps)*[drug])
        res['CCL'].extend(len(reps)*[ccl])
        
        drug_to_delta[drug]=np.mean(_t['Delta'].values)

res = pd.DataFrame(res)
res = res[res['Drug'].isin(drugs)]
res = res.dropna()
sns.scatterplot(res,x='AAC',y='Drug',style='CCL',hue='CCL',s=100)
# plt.axvspan(0.1-delta/2, 0.1+delta/2,alpha = 0.2, color = 'k')
# plt.axvline(0.1,color = 'k', linestyle='--')
plt.tight_layout()
plt.savefig("../figs/fig1/noise_fig.png",dpi=500)
plt.close()
sns.set_theme(rc={'figure.figsize':(10,3)})



for drug in ['Erlotinib', 'Gemcitabine']:
    print(drug)
    res1 = res[res['Drug'].isin([drug])]
    print(res1)

    sns.scatterplot(res1,x='AAC',y='Drug',style = 'CCL',hue='CCL',legend=False,s=500)

    for drug in pd.unique(res1['Drug']):
        _t = res[(res['Drug']==drug) & (res['CCL']=="A-549")]
       
        ls = "--"
        clr = 'purple'
        
        est = np.median(_t['AAC'])
        plt.axvline(est,color = clr, linestyle=ls)
        
        plt.axvspan(est-delta/2, est+delta/2,alpha = 0.2, color = 'k')
        # plt.xlim(0,0.2)

        _t = res[(res['Drug']==drug) & (res['CCL']=="CAL-51")]
        est = np.median(_t['AAC'])
        ls = "--"
        clr = 'orange'
        plt.axvline(est,color = clr, linestyle=ls)

        plt.ylabel(None)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        plt.savefig(f"../figs/fig1/subdelta_{drug}.png",dpi=500)
        plt.close()

# res = res[res['Drug'].isin(['Erlotinib','Gefitinib'])]
# drug_to_delta['Erlotinib']

# sns.scatterplot(res,x='AAC',y='Drug',hue='Drug',legend=False)
# for drug in pd.unique(res['Drug']):
#     _t = res[res['Drug']==drug]
#     delta = drug_to_delta[drug]
#     if drug == 'Erlotinib':
#         ls= '-'
#     else:
#         ls = '--'
#     plt.axvline(np.median(_t['AAC']+delta/2), color='k',linestyle=ls)
#     plt.axvline(np.median(_t['AAC']-delta/2),color = 'k', linestyle=ls)
# plt.tight_layout()
# plt.savefig("../figs/egfr_delta.png",dpi=500)

# plt.close()

# # keeps = sorted([v for v in drug_to_delta.values()])[:5]
# # points=[k for k in drug_to_delta if drug_to_delta[k] in (keeps)]
# # print(keeps)
# # print(points)
# # # for k in points:

# # print([drug_to_list[k] for k in points])
# # # print([np.mean(drug_to_list[k]) for k in points])
