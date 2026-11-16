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


benchmarks = pd.read_csv("../results/benchmarks/kfold/scalar_metrics_ci.csv")
print(benchmarks)


benchmarks = benchmarks[benchmarks['Eval. Type']!='All']
drugs = pd.unique(benchmarks['Drug'])

drugs = utils.read_text_file("../procdata/drug_lists/post_qc.txt")
# drugs = [x for x in drugs if x!='JQ1 compound']
all_neur = pd.DataFrame()
all_models = pd.DataFrame()
neur_gaps = pd.DataFrame()
for drug in drugs:
	ecinet = pd.read_csv(f"../results/neurals/kfold/{drug}/ECINET/scalar_metrics.csv")
	# ecinet = pd.read_csv(f"../results/neurals/kfold/{drug}/ECINET/param_search.csv")
	
	# sys.exit()
	one_narrow = pd.read_csv(f"../results/neurals/kfold/{drug}/ONE_NARROW/scalar_metrics.csv")
	two_narrow = pd.read_csv(f"../results/neurals/kfold/{drug}/TWO_NARROW/scalar_metrics.csv")
	one_narrow['Model'] = "One Layer"
	two_narrow['Model'] = "Two Layer"
	
	# one_wide = pd.read_csv(f"../results/neurals/kfold/{drug}/ONE_WIDE/scalar_metrics.csv")
	# two_wide = pd.read_csv(f"../results/neurals/kfold/{drug}/TWO_WIDE/scalar_metrics.csv")
	
	neurals = pd.concat( (ecinet,one_narrow,two_narrow))
	
	neurals = neurals[( neurals['Eval. Type']!='All') & (neurals['cv_score']=='ci')]
	drug_bench = benchmarks[(benchmarks['Drug']==drug)]


	neurals = neurals[['Drug','Data Set','Eval. Type','Model','Pearson','Delta', 'Concordance']]
	drug_bench = drug_bench[['Drug','Data Set','Eval. Type','Model','Pearson','Delta','Concordance']]
	drug_all = pd.concat((neurals,drug_bench))
	drug_all.drop_duplicates(inplace=True)
	print(drug_all.head())
	drug_all['Concordance'] = np.round(drug_all['Concordance'].values,3)
	drug_all['Pearson'] = np.round(drug_all['Pearson'].values,3)
	all_models = pd.concat((all_models,drug_all))
	for ds in ['gCSI','GDSC2']:
		for et in ['Seen','Unseen']:
			_temp = drug_all[(drug_all['Data Set']==ds) & (drug_all['Eval. Type']==et)]

			for metric in ['Pearson', 'Concordance']:
				ax = sns.barplot(data=_temp, x='Model', y=metric,hue='Model',errorbar=None)
				for bars in ax.containers:
					ax.bar_label(bars, fontsize=10);
				
				plt.title(f"{drug}\n{et} Cell Lines in {ds}")
				if metric=='Concordance':
					plt.ylim(0.5,1)
				plt.xticks(rotation=30)
				plt.tight_layout()
				# plt.show()
				os.makedirs(f"../figs/neurals/{drug}/",exist_ok=True)
				plt.savefig(f"../figs/neurals/{drug}/{metric}_{ds}_{et}.png")
				plt.close()
				# plt.show()


for drug in drugs:
	for ds in ['gCSI','GDSC2']:
		for et in ['Seen','Unseen']:
			temp = all_models[
				(all_models['Drug']==drug) &
				(all_models['Data Set']==ds) &
				(all_models['Eval. Type']==et)]

			best

	delta_count = pd.DataFrame(all_models[all_models['Model']==model]['Delta'].value_counts())
	print(model)
	print(delta_count)
# 	sys.exit()
# 	# neurals = neurals[['Drug','Data Set','Eval. Type','Model','Pearson','Spearman Distance','Scaled Spearman','Concordance']]
# 	# neurals.drop_duplicates(inplace=True)

# 	ecn = ecn[['Drug','Data Set','Eval. Type','Model','Pearson','Spearman Distance','Scaled Spearman','Concordance','rCI']]
# 	ecn.drop_duplicates(inplace=True)
	
	
# 	# sys.exit()# search = pd.read_csv(f"../results/neurals/kfold/{drug}/ECINET/param_search.csv")
# 	# search = search[search['cv_score']=='ci']
	

# 	 # & (benchmarks['Gap']==d)]
# 	print('db1')
# 	print(drug_bench)

# 	drug_bench = drug_bench[['Drug','Data Set','Eval. Type','Model','Pearson','Spearman Distance','Scaled Spearman','Concordance','rCI']]
# 	drug_bench.drop_duplicates(inplace=True)
	
# 	# print(ecn)
# 	# print('db2')
# 	# print(drug_bench)
# 	# sys.exit()
# 	data = pd.concat((drug_bench,ecn))
	

# 	data = data[data['Eval. Type']!='All']
# 	for ds in ['GDSC2','gCSI']:
# 		for et in ['Seen','Unseen']:
# 			temp = data[data['Data Set']==ds]
# 			temp = temp[temp['Eval. Type']==et]
# 			print(temp)

# 	# print(data)
# sys.exit()
	
# 	# neurals = pd.concat((ecinet,one_narrow,two_narrow))
# 	# neurals = neurals[neurals['Eval. Type']!='All']
# 	# print(neurals)
# 	# sys.exit()
# 	for ds in ['GDSC2','gCSI']:
# 		for et in ['Seen','Unseen']:
# 			temp = drug_bench[(drug_bench['Data Set']==ds) & (drug_bench['Eval. Type']==et)]
# 			# print(temp)
# 			bench_pearson = np.max(temp['Pearson'].values)
# 			bench_ci =np.max(temp['Concordance'].values)
# 			# print(bench_pearson)
# 			# print(bench_ci)
# 			temp_n = neurals[(neurals['Data Set']==ds) & (neurals['Eval. Type']==et)]

# 			# print(temp_n)
# 			temp_n['Concordance'] = temp_n['Concordance']-bench_ci
# 			temp_n['Pearson'] = temp_n['Pearson']-bench_pearson
# 			temp_n['Drug']= drug
# 			# print(temp_n)
# 			neur_gaps = pd.concat((neur_gaps,temp_n))
# 		# sys.exit()


# 	all_neur = pd.concat((all_neur,neurals))

# 	data = pd.concat((drug_bench,neurals))
# 	all_models = pd.concat((all_models, data))
# 	data = data[data['Model'].isin(['ElasticNet','RandomForest','ECINET','One Layer', 'Two Layer'])]
# 	# print(data)
# 	# sys.exit()
# 	p = sns.catplot(neurals,kind = 'bar',x='Model',y='Concordance', row = 'Eval. Type', col = "Data Set",hue='Model')
# 	# ax.set(ylim=(0.5,0.8))
# 	p.fig.suptitle(f"{drug}")
# 	plt.close()
# 	# sys.exit()


# for ds in ['GDSC2','gCSI']:
# 	for et in ['Seen','Unseen']:
# 		temp = neur_gaps[(neur_gaps['Data Set']==ds) & (neur_gaps['Eval. Type']==et)]
		
# 		sns.barplot(temp, y='Drug', x='Concordance',hue='Model')
		
# 		plt.title(f"Neural Concordance net of Best Benchmark \n Evaluated on {et} cell lines in {ds}")
# 		plt.tight_layout()
# 		plt.savefig(f"Concordance_net_{et}_{ds}.png")
# 		plt.close()
# 		sns.barplot(temp, y='Drug', x='Pearson',hue='Model')
		
# 		plt.title(f"Neural Pearson net of Best Benchmark \n Evaluated on {et} cell lines in {ds}")
# 		plt.tight_layout()
# 		plt.savefig(f"Pearson_net_{et}_{ds}.png")
# 		plt.close()
# sys.exit()

# print(all_neur)
# all_models = all_models[all_models['Model'].isin(['ElasticNet','RandomForest','ECINET','ONE_WIDE'])]
# for ds in ['GDSC2','gCSI']:
# 	for et in ['Seen','Unseen']:
# 		temp = all_models[(all_models['Data Set']==ds) & (all_models['Eval. Type']==et)]
# 		print(temp)
# 		sns.violinplot(temp,x='Model',y='Concordance',hue='Model')
# 		plt.title(f"Concordance in {ds} with {et} Cell Lines")
# 		plt.savefig(f"CINET_concordance_{ds}_{et}.png")
# 		plt.close()
# 		sns.violinplot(temp,x='Model',y='Pearson',hue='Model')
# 		plt.title(f"Correlation in {ds} with {et} Cell Lines")
# 		plt.savefig(f"CINET_corr_{ds}_{et}.png")
# 		plt.close()
# sys.exit()
	
# 	# fig_dir = f"../figs/{target_data}/{ccl_type}/"
	
# 	# plotting_ks = [1,5,10, 15,20]
	
# 	# os.makedirs(fig_dir,exist_ok=True)


# 	# ranking_benchmarks_f = f"../results/{target_data}/{ccl_type}/ranking_metrics.csv"
# 	# ranking_benchmarks = pd.read_csv(ranking_benchmarks_f)

# 	# ranking_neural_f = f"../results/neurals/{target_data}/BCE/ranking_metrics.csv"

# 	# ranking_neural = pd.read_csv(ranking_neural_f)
# 	# neural_models = pd.unique(ranking_neural['Model'])

# 	# # print(ranking_benchmarks.shape)
# 	# # print(ranking_neural.shape)

# 	# # print(ranking_neural.columns)
# 	# # print(ranking_benchmarks.columns)
# 	# # sys.exit()


# 	# ranking_df = pd.concat( 
# 	# 	(ranking_benchmarks[['Drug', 'Eval. Type', 'Model', 'Perm Overlap','NDCG0','K','Delta', 'Tau']],
# 	# 		ranking_neural[['Drug', 'Eval. Type', 'Model',  'Perm Overlap','NDCG0','K','Delta', 'Tau']]),
# 	# 	axis = 0
# 	# 	)

# 	# print(ranking_df.shape)
# 	# sys.exit()
# 	# spearman_df.drop_duplicates(inplace=True)
# 	# print(spearman_df.shape)

# 	# print(spearman_df)
# 	# sub = spearman_df[spearman_df['Model'].isin(benchmark_models)]
# 	# sub.drop_duplicates(inplace= True)
# 	# print(spearman_df)
# 	# hlines = {x:y for x,y in zip(sub['Model'], sub['Scaled Spearman'])}
	

# 	# for drug in pd.unique(spearman_df['Drug']):
# 	# 	for et in pd.unique(spearman_df['Eval. Type']):
# 	# 		temp = spearman_df[(spearman_df['Drug']==drug) & (spearman_df['Eval. Type']==et)]
# 	# 		nn = temp[temp['Model'].isin(neural_models)]
# 	# 		sub = temp[temp['Model'].isin(benchmark_models)]
# 	# 		sub.drop_duplicates(inplace= True)
# 	# 		hlines = {x:y for x,y in zip(sub['Model'], sub['Scaled Spearman'])}
# 	# 		for tau in pd.unique(nn['Tau']):
# 	# 			_n = nn[nn['Tau']==tau]
# 	# 			ax = sns.scatterplot(_n, x = 'Delta', y='Scaled Spearman', hue = 'Model')
# 	# 			print(hlines)
# 	# 			for hl in hlines.keys():
# 	# 				if hl == 'ElasticNet':
# 	# 					ax.axhline(hlines[hl],label = hl)
# 	# 			plt.title(f"{drug} Scaled Spearman {et} Tau = {tau}")
# 	# 			plt.show()
# 	# 	sys.exit()





































# 	# sys.exit()
# 	# scalar_benchmarks_f = f"../results/{target_data}/{ccl_type}/scalar_metrics.csv"
# 	# scalar_benchmarks = pd.read_csv(scalar_benchmarks_f)

# 	# scalar_neural_f = f"../results/neurals/{target_data}/BCE/scalar_metrics.csv"

# 	# scalar_neural = pd.read_csv(scalar_neural_f)
# 	# neural_models = pd.unique(scalar_neural['Model'])


# 	# print(neural_models)


# 	# t = scalar_benchmarks[['Drug','Model','Eval. Type', 'Pearson','Concordance','Delta','Tau']]
# 	# p = scalar_neural[['Drug', 'Model', 'Eval. Type', 'Pearson','Concordance','Delta','Tau']]
# 	# t.drop_duplicates(inplace= True)
# 	# p.drop_duplicates(inplace= True)
# 	# t = pd.concat((t,p),axis = 0)

	
	
# 	# # print(_t.head())
# 	# # for e in pd.unique(t['Eval. Type']):
# 	# for e in ['All','Unseen','Seen']:	
# 	# 	for drug in pd.unique(t['Drug']):
# 	# 		for metric in ['Concordance', 'Pearson']:
			
# 	# 			_t = t[(t['Eval. Type']==e) & (t['Drug']==drug)]

# 	# 			sub = _t[_t['Model'].isin(benchmark_models)]
# 	# 			sub.drop_duplicates(inplace= True)

# 	# 			print(sub.shape)
# 	# 			print(sub)
# 	# 			hlines = {x:y for x,y in zip(sub['Model'], sub[metric])}
# 	# 			print(hlines)
				
# 	# 			nn = _t[_t['Model'].isin(neural_models)]
# 	# 			print(nn)

# 	# 			for tau in pd.unique(nn['Tau']):
# 	# 				temp = nn[nn['Tau']==tau]
					
# 	# 				ax = sns.scatterplot(temp, x = 'Delta', y=metric, hue = 'Model')
# 	# 				if metric == 'Concordance':
# 	# 					ax.set(ylim=(0.5,0.8))
# 	# 				for hl in hlines.keys():
# 	# 					if hl == 'ElasticNet':
# 	# 						ax.axhline(hlines[hl],label = hl)
# 	# 				plt.title(f"{drug} {metric} {e} Tau = {tau}")
# 	# 				plt.show()
# 	# 		sys.exit()

# 	# 			# # baseline_levels = 
# 	# 			# # _t['Concordance'] = _t['Concordance']-0.5
# 	# 			# ax = sns.catplot(data = _t,kind='bar', y = 'Model',x = 'Pearson',hue = 'Model')
# 	# 			# # ax.set(xlim = (0.5,1.2))
# 	# 			# # print(z)
# 	# 			# plt.title(f"Correlation for {drug} in {e} Cell Lines")
# 	# 			# plt.tight_layout()
# 	# 			# plt.show()
# 	# 			# sys.exit()
# 	# 		# if c>=5:
# 	# 		# 	sys.exit()

	
# 	# 	
# import pandas as pd
# import numpy as np 
# import matplotlib.pyplot as plt 
# import seaborn as sns
# import sys
# from collections import defaultdict
# import os




# ###
# #
# #	All Pairwise Deltas
# #
# ###

# with open("../procdata/drug_lists/feasible_list.txt", "r") as istream:
# 	drugs = istream.readlines()
# 	drugs = [x.rstrip() for x in drugs]

# models = ['ECINET','ONE_LAYER','TWO_LAYER','PARS_PROJ']
# # broad_effect = ['Docetaxel', 'Paclitaxel', 'Vincristine', 'Gemcitabine', 'Bortezomib', 'AZD7762', 'Ibrutinib']
# # drugs = ['Docetaxel','Paclitaxel']
# # drugs = broad_effect
# # drugs = ['Paclitaxel', 'Gefitinib', 'Erlotinib', 'Lapatinib', 'Docetaxel']
# scalar_results = pd.DataFrame()
# ranking_results = pd.DataFrame()

# for drug in drugs:
# 	print(drug)
# 	print("------")
# 	print("scalar")
# 	print("------")
	
# 	temp = pd.read_csv(f"../results/{drug}/neurals/scalar_metrics.csv")
# 	print(temp.shape)
# 	# print(temp.shape)
# 	temp['BinnedTau'] = [tau if tau in [0,0.1] else '3Q' for tau in temp['Tau'].values]
# 	temp = temp[temp['BinnedTau'].isin([0., '3Q'])]
# 	temp = temp[temp['Model'].isin(models)]
# 	# temp['Model'] = [m if m in ['ECINET','ONE_LAYER','TWO_LAYER'] else 'Denoise' for m in temp['Model'].values]
# 	scalar_results = pd.concat((scalar_results,temp),axis=0)

# 	print(temp['Tau'].value_counts())
# 	print(temp['BinnedTau'].value_counts())

# 	temp = pd.read_csv(f"../results/{drug}/neurals/ranking_metrics.csv")
# 	print("------")
# 	print("ranking")
# 	print("------")
# 	print(temp.shape)
# 	temp = temp[["Drug","Data Set","Eval. Type","Model","Perm Overlap","NDCG",	"K",	"Delta",	"Tau"]]
# 	temp['BinnedTau'] = [tau if tau in [0,0.1] else '3Q' for tau in temp['Tau'].values]
# 	temp = temp[temp['BinnedTau'].isin([0., '3Q'])]
# 	temp = temp[temp['Model'].isin(models)]
# 	# temp['Model'] = [m if m in ['ECINET','ONE_LAYER','TWO_LAYER'] else 'Denoise' for m in temp['Model'].values]
# 	ranking_results = pd.concat((ranking_results,temp),axis=0)
	
# 	print(temp['Tau'].value_counts())
# 	print(temp['BinnedTau'].value_counts())

# # scalar_results['Concordance'] = [1-c if c<0.5 else c for c in scalar_results['Concordance'].values]
# # plt.show()
# # sys.exit()




# # for model in models:
# # 	print(model)
# # 	plotting_df = scalar_results[
# # 		(scalar_results['Model']==model) & 
# # 		(scalar_results['BinnedTau'].isin(['3Q'])) & (scalar_results['Gap']==0.2)
# # 		]
# # 	g = sns.FacetGrid(plotting_df, row = "Eval. Type", col = "Data Set",hue = 'BinnedTau',ylim = (0.40,1.))
# # # p = sns.boxplot(plotting_df, x = 'Delta', y = 'Concordance', hue = 'BinnedTau')
# # 	
# # 	# g.set_title(f"All Models")
# # 	plt.show()
# # sys.exit()
# # models = ['ECINET','ONE_LAYER','TWO_LAYER','Denoise']

# counting_s = scalar_results[scalar_results['BinnedTau']==0]

# best_deltas = defaultdict(list)
# for drug in drugs:
# 	for model in models:
# 		for val_set in ['gCSI','GDSC2']:
# 			for et in ['Seen', 'Unseen', 'All']:
# 				temp = counting_s[ (counting_s['Model']==model) &
# 					   (counting_s['Drug']==drug) &
# 					   (counting_s['Data Set']==val_set) &
# 					   (counting_s['Eval. Type']==et)
# 					   ]
# 				for metric in ['Concordance','Pearson']:
# 					m = np.max(temp[metric].values)
# 					am = np.argmax(temp[metric].values)
# 					d = temp['Delta'].values[am]
					
# 					best_deltas['Drug'].append(drug)
# 					best_deltas['Model'].append(model)
# 					best_deltas['Eval. Set'].append(val_set)
# 					best_deltas['Eval. Type'].append(et)
# 					best_deltas['Metric'].append(metric)
# 					best_deltas['Best Delta'].append(d)

# print("\n\n\n\n\n\n\n")
# best_deltas = pd.DataFrame(best_deltas)
# for metric in ['Concordance','Pearson']:
# 	print(f"All - {metric}")
# 	print(best_deltas[best_deltas['Metric']==metric]['Best Delta'].value_counts())			

# # print("All")
# # print(best_deltas['Best Delta'].value_counts())			
# # print("\n\n\n\n\n\n\n")




# best_taus = defaultdict(list)
# for drug in drugs:
# 	for model in models:
# 		for val_set in ['gCSI','GDSC2']:
# 			for et in ['Seen', 'Unseen', 'All']:
# 				for delta in pd.unique(scalar_results['Delta']):
# 					temp = scalar_results[ (scalar_results['Model']==model) &
# 						   (scalar_results['Drug']==drug) &
# 					   	(scalar_results['Data Set']==val_set) &
# 					   	(scalar_results['Eval. Type']==et) &
# 					   	(scalar_results['Delta']==delta) 
# 					   	]
				
# 				for metric in ['Concordance','Pearson']:
# 					m = np.max(temp[metric].values)
# 					am = np.argmax(temp[metric].values)
# 					d = temp['BinnedTau'].values[am]
					
# 					best_taus['Drug'].append(drug)
# 					best_taus['Model'].append(model)
# 					best_taus['Eval. Set'].append(val_set)
# 					best_taus['Eval. Type'].append(et)
# 					best_taus['Metric'].append(metric)
# 					best_taus['Delta'].append(delta)
# 					best_taus['Best Tau'].append(d)

# print("\n\n\n\n\n\n\n")
# best_taus = pd.DataFrame(best_taus)
# for metric in ['Concordance','Pearson']:
# 	print(f"All - {metric}")
# 	print(best_taus[best_taus['Metric']==metric]['Best Tau'].value_counts())			

# print("All")
# print(best_taus['Best Tau'].value_counts())			
# print("\n\n\n\n\n\n\n")


# for et in ['Seen', 'Unseen', 'All']:
# 	temp = best_taus[best_taus['Eval. Type']==et]
# 	print(et)
# 	for metric in ['Concordance','Pearson']:
# 		print(f"All - {metric}")
# 		print(temp[temp['Metric']==metric]['Best Tau'].value_counts())			

# 	print("All")
# 	print(temp['Best Tau'].value_counts())		

# cd = [0., 0.1, 0.15, 0.2]

# best_taus = best_taus[best_taus['Delta'].isin(cd)]

# print("restricting")
# best_taus = pd.DataFrame(best_taus)
# for metric in ['Concordance','Pearson']:
# 	print(f"All - {metric}")
# 	print(best_taus[best_taus['Metric']==metric]['Best Tau'].value_counts())			

# print("All")
# print(best_taus['Best Tau'].value_counts())
# sys.exit()			
# for val_set in ['gCSI','GDSC2']:
# 	for et in ['Seen','Unseen','All']:
		
# 			print(f"{val_set} - {et} - {metric}")
# 			temp = best_deltas[(best_deltas['Eval. Set']==val_set) & (best_deltas['Eval. Type']==et) & (best_deltas['Metric']==metric)]
# 			print(temp['Best Delta'].value_counts())
# 			# sns.histplot(temp, x= 'Best Delta', binwidth = 0.005)





# # iterate and over the models, drugs, testing screens, and eval types. 
# # and enumerate:
# #			which delta had the best perf. 


# for model in models:
# 	for val_set in ['gCSI','GDSC2']:
# 		for metric in ['Concordance','Pearson']:
# 			plotting_df = scalar_results[
# 				(scalar_results['Model']==model) &
# 				(scalar_results['Data Set']==val_set) 
# 				]
# 			# g = sns.FacetGrid(plotting_df, col = "Eval. Type",hue = 'BinnedTau')
# 			# g.map(sns.boxplot, 'Delta', 'Concordance', hue_order = [0, '3Q'])
# 			# g.add_legend()
# 			p = sns.catplot(data= plotting_df, x = 'Delta', y = metric, hue = 'BinnedTau', col = 'Eval. Type', kind = 'box')
# 			plt.suptitle(f"{model} - {metric} vs. Delta \n {val_set} All Drug")
# 			if metric == "Concordance":
# 				p.set(ylim=(0.5, 1))

# 			plt.tight_layout()
# 			sns.move_legend(p, "upper right")
# 			plt.savefig(f"../figs/{model}_{val_set}_{metric}.png")
# 			plt.close()
# sys.exit()


# import pandas as pd
# import numpy as np 
# import matplotlib.pyplot as plt 
# import seaborn as sns
# import sys
# from collections import defaultdict
# import os




# ###
# #
# #	All Pairwise Deltas
# #
# ###

# with open("../procdata/drug_lists/feasible_list.txt", "r") as istream:
# 	drugs = istream.readlines()
# 	drugs = [x.rstrip() for x in drugs]


# plot_ks = [3,5,10]

# benchmark_models = ['RandomForest', 'ElasticNet','Univariable', 'UniLasso','GroundTruth','KRR']

# models = ['ECINET','ONE_LAYER','TWO_LAYER','PARS_PROJ']
# models = ['ONE_LAYER'] 
# scalar_benchmarks = pd.read_csv("../results/benchmarks/scalar_metrics.csv")
# ranking_benchmarks = pd.read_csv("../results/benchmarks/ranking_metrics.csv")

# print(ranking_benchmarks.head())
# print(ranking_benchmarks.columns)

# scalar_benchmarks = scalar_benchmarks[scalar_benchmarks['Model'].isin(benchmark_models)]

# scalar_cols = ['Drug', 'Data Set','Eval. Type', 'Model', 'Pearson', 'Concordance', 'Delta', 'Tau']
# scalar_sub = ['Drug', 'Data Set','Eval. Type', 'Model', 'Pearson', 'Concordance']

# rank_cols = ['Drug', 'Data Set', 'Eval. Type', 'Model', 'Perm Overlap', 'NDCG', 'K', 'Delta', 'Tau']
# rank_sub = ['Drug', 'Data Set', 'Eval. Type', 'Model', 'Perm Overlap', 'NDCG', 'K']
# deltas = [0.2]
# metric = "Concordance"


# # metric = 'NDCG'
# # for drug in drugs:
	
# # 	_rb = ranking_benchmarks[ranking_benchmarks['Drug']==drug]
# # 	_rb = _rb[_rb['Model'].isin(benchmark_models)]
# # 	_rb = _rb[_rb['Eval. Type'].isin(['Seen','Unseen'])]
# # 	_rn = pd.read_csv(f"../results/{drug}/neurals/ranking_metrics.csv")
# # 	_rb = _rb[rank_cols]
# # 	_rb.drop_duplicates(inplace=True)

# # 	_rn = _rn[rank_cols]
# # 	_rn = _rn[_rn['Delta'].isin(deltas)]
# # 	_rn = _rn[_rn['Model'].isin(models)]
# # 	_rn.drop_duplicates(inplace= True)
# # 	_rn = _rn[_rn['Eval. Type'].isin(['Seen','Unseen'])]
# # 	_rn['BinnedTau'] = [tau if tau in [0,0.1] else '3Q' for tau in _rn['Tau'].values]
# # 	_rn = _rn[_rn['BinnedTau'].isin([0,'3Q'])]
	

# # 	_rn = _rn[_rn['K'].isin(plot_ks)]
# # 	_rb = _rb[_rb['K'].isin(plot_ks)]
	

	
# # 	# plotting_df = pd.concat((_sn, _sb))
	
# # 	_temp = _rn[_rn['BinnedTau']==0.]
# # 	plotting_df = pd.concat((_temp[rank_cols],_rb[rank_cols]))
# # 	plotting_df = plotting_df[plotting_df['Data Set']=='gCSI']
# # 	print(plotting_df.head())
# # 	print(_rb.head())
# # 	p = sns.catplot(data= plotting_df, x = 'K', y = metric, hue = 'Model',
# # 				col = 'Eval. Type', kind = 'bar')
# # 	plt.suptitle(f"{drug} -{metric}")
	
# # 	plt.show()
# # sys.exit()	
# # 	# print(_rn.head())
# # 	# sys.exit()
# # 	# # _sn['Model'] = [f"{x}-{y}-{z}" for x,y,z in zip(_sn['Model'].values, _sn['Delta'].values, _sn['BinnedTau'].values)]
# # 	# print(_sn.head())
	

# # 	# _sb = _sb[scalar_sub]
# # 	# _sn = _sn[scalar_sub]	

	

# # 	# p = sns.catplot(data= plotting_df, y = 'Model', x = metric,
# # 	# 	row = 'Data Set', col = 'Eval. Type', kind = 'bar')
# # 	# plt.suptitle(f"{drug} -{metric}")
# # 	# plt.tight_layout()
# # 	# plt.show()
# # 	# # sys.exit()
	
# # sys.exit()

# ### SCALARS

# best_neurals = pd.DataFrame()

# for drug in drugs:
	
# 	_sb = scalar_benchmarks[scalar_benchmarks['Drug']==drug]
# 	_sb = _sb[_sb['Eval. Type'].isin(['Seen','Unseen'])]
# 	_sn = pd.read_csv(f"../results/{drug}/neurals/scalar_metrics.csv")
# 	_sn = _sn[_sn['Model'].isin(models)]
# 	_sb = _sb[scalar_cols]
# 	_sn['BinnedTau'] = [tau if tau in [0,0.1] else '3Q' for tau in _sn['Tau'].values]
# 	_sn = _sn[_sn['BinnedTau'].isin([0,'3Q'])]
# 	for ds in ['gCSI','GDSC2']:
# 		for et in ['Seen','Unseen']:
# 			_tn = _sn[
# 				(_sn['Data Set']==ds) &
# 				(_sn['Eval. Type']==et)
# 				]
# 			_tb = _sb[
# 				(_sb['Data Set']==ds) &
# 				(_sb['Eval. Type']==et)
# 				]
# 			for metric in ['Concordance','Pearson']:
			
				
# 				idx = np.argmax(_tn[metric].values)
# 				#	 print(pd.DataFrame(_sn.iloc[[idx]]))
# 				_tau = _tn.iloc[[idx]]['BinnedTau'].values[0]
# 				_d = _tn.iloc[[idx]]['Delta'].values[0]

# 				bn = _tn.iloc[[idx]]
# 				bn['Model'] = "Best Neural"
# 				best_neurals = pd.concat((best_neurals,bn))
# 			# plotting_df = pd.concat(
# 			# 	(_tb[scalar_sub], _tn[scalar_sub].iloc[[idx]])
# 			# 	)
			
# 			# plotting_df.drop_duplicates(inplace=True)
# 			# plotting_df[metric] = np.round(plotting_df[metric],3)
# 			# print(plotting_df.head(10))
# 			# sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
# 			# ax = sns.barplot(data= plotting_df, y = metric, x = 'Model',hue = 'Model')
# 			# for container in ax.containers:
# 			# 	ax.bar_label(container)
# 			# plt.title(f"For {drug} - {ds} - {et}\ndelta= {_d}. tau = {_tau}")
# 			# plt.tight_layout()
# 			# plt.savefig(f"{drug}_{metric}.png")
# 			# # plt.show()
# 			# plt.close()

# sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
# _sb = scalar_benchmarks[scalar_benchmarks['Eval. Type'].isin(['Seen','Unseen'])]
# _sb = _sb[scalar_sub]
# _sb.drop_duplicates(inplace=True)
# best_neurals = best_neurals[scalar_sub]
# print(_sb.head())
# plotting_df = pd.concat((_sb,best_neurals))
# print(plotting_df.head())
# ax = sns.catplot(data = plotting_df, x = 'Model', y = 'Pearson', 
# 	legend = 'brief',row = 'Data Set', col = 'Eval. Type', hue = 'Model', kind = 'violin')
# # plt.tight_layout() 
# ax.set_xticklabels(rotation=30)
# plt.savefig(f"Pearson_Comparison.png")
# plt.close()

# res = defaultdict(list)
# for drug in drugs:
# 	for ds in ['gCSI','GDSC2']:
# 		for et in ['Seen','Unseen']:
# 			# for metric in ['Concordance','Pearson']:
# 			temp = plotting_df[
# 				(plotting_df['Drug']==drug) &
# 				(plotting_df['Data Set']==ds) &
# 				(plotting_df['Eval. Type']==et)
# 				]
			
# 			bn = temp[temp['Model']=="Best Neural"]['Pearson'].values[0]
# 			bb = max(temp[temp['Model'].isin(benchmark_models)]['Pearson'].values)
# 			res['Drug'].append(drug)
# 			res['Data Set'].append(ds)
# 			res['Eval. Type'].append(et)
# 			res['Diff'].append(bn-bb)

# res = pd.DataFrame(res)
# # print(res.head())
# ax = sns.catplot(data = res,y = 'Drug', x = 'Diff', row = 'Data Set', col = 'Eval. Type', hue = 'Drug', kind = 'bar')
# plt.savefig("Pearson_Difference.png")
# plt.show()
# sys.exit()
			



# for drug in drugs:
	
# 	_sb = scalar_benchmarks[scalar_benchmarks['Drug']==drug]
# 	_sb = _sb[_sb['Eval. Type'].isin(['Seen','Unseen'])]
# 	_sn = pd.read_csv(f"../results/{drug}/neurals/scalar_metrics.csv")
# 	_sb = _sb[scalar_cols]
# 	_sb.drop_duplicates(inplace=True)
# 	_sn = _sn[scalar_cols]
# 	_sn = _sn[_sn['Delta'].isin(deltas)]
# 	_sn = _sn[_sn['Model'].isin(models)]
# 	_sn.drop_duplicates(inplace= True)
# 	_sn = _sn[_sn['Eval. Type'].isin(['Seen','Unseen'])]
# 	_sn['BinnedTau'] = [tau if tau in [0,0.1] else '3Q' for tau in _sn['Tau'].values]
# 	_sn = _sn[_sn['BinnedTau'].isin([0,'3Q'])]
	
# 	print(_sn.head())
# 	# _sn['Model'] = [f"{x}-{y}-{z}" for x,y,z in zip(_sn['Model'].values, _sn['Delta'].values, _sn['BinnedTau'].values)]
# 	print(_sn.head())
	

# 	_sb = _sb[scalar_sub]
# 	_sn = _sn[scalar_sub]

# 	plotting_df = pd.concat((_sn, _sb))
# 	plotting_df = plotting_df[plotting_df['Data Set']=='GDSC2']


# 	p = sns.catplot(data= plotting_df, y = 'Model', x = metric,
# 		row = 'Data Set', col = 'Eval. Type', kind = 'bar')
# 	plt.suptitle(f"{drug} -{metric}")
# 	# plt.tight_layout()
# 	plt.show()
# 	# sys.exit()
# sys.exit()