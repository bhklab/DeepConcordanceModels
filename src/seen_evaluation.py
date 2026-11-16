###
#
#  Script to compute model performance across several metrics
#  as the model presently only returns predictions
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
from collections import defaultdict
from scipy.stats import pearsonr
import tqdm
import matplotlib.patches as mpatches



baseline_scalars = pd.read_csv("../results/baseline_scalars.csv",index_col=0)
baseline_ranking = pd.read_csv("../results/baseline_ranking.csv",index_col=0)

neural_scalars =  pd.read_csv("../results/neurals/scalars.csv",index_col=0)
neural_ranking = pd.read_csv("../results/neurals/ranking.csv",index_col=0)

neural_ranking = neural_ranking[neural_ranking['Model']=='Two Layer']
neural_scalars = neural_scalars[neural_scalars['Model']=='Two Layer']

neural_ranking['Model'] = ["DeepCINET-"+str(x) for x in neural_ranking['Percentile'].values]
neural_scalars['Model'] = ["DeepCINET-"+str(x) for x in neural_scalars['Percentile'].values]
neural_scalars = neural_scalars[neural_scalars['Percentile'].isin([0,50,95])]
neural_ranking = neural_ranking[neural_ranking['Percentile'].isin([0,50,95])]

baseline_ranking = baseline_ranking[baseline_ranking['K'].isin([5,10])]

neural_ranking = neural_ranking[neural_ranking['K'].isin([5,10])]

sizes = pd.read_csv("../results/dataset_sizes.csv",index_col=0)
seen_cutoff = 20
fig_dir = "../figs/fig2/"
os.makedirs(fig_dir,exist_ok=True)

drugs = utils.read_text_file("../procdata/drug_lists/post_qc.txt")

print(drugs)
print(len(drugs))

scalar_res, rank_res = [],[]
rel_rank = defaultdict(list)


# screens = ['GDSC1','GDSC2','gCSI']
screens = ['GDSC1']
for screen in screens:
    
    
    for drug in drugs:
        for et in ['Unseen','Seen']:
            
            
            # subset neural data
            _ns = neural_scalars[(neural_scalars['Drug']==drug) & 
                                 (neural_scalars['Screen']==screen) &
                                 (neural_scalars['Eval Type']==et)
                                 ][['Model','Drug','Screen','Pearson','Concordance','Eval Type']]
            
            _nr = neural_ranking[(neural_ranking['Drug']==drug) & 
                                 (neural_ranking['Screen']==screen) &
                                 (neural_ranking['Eval Type']==et)
                                 ][['Model','Drug','Screen','K','NDCG','Perm Overlap', 'Eval Type']]
            
            # subset baselines

            _bs = baseline_scalars[(baseline_scalars['Drug']==drug) & 
                                 (baseline_scalars['Screen']==screen) &
                                 (baseline_scalars['Eval Type']==et)
                                 ][['Model','Drug','Screen','Pearson','Concordance','Eval Type']]
            
            
            _br = baseline_ranking[(baseline_ranking['Drug']==drug) & 
                                 (baseline_ranking['Screen']==screen) &
                                 (baseline_ranking['Eval Type']==et)
                                 ][['Model','Drug','Screen','K','NDCG','Perm Overlap', 'Eval Type']]
            

            _r = pd.concat([_nr,_br])
            _s = pd.concat([_ns,_bs])
            
            scalar_res.append(_s)
            rank_res.append(_r)
            
            
            for metric in ['Pearson','Concordance']:
               
                z = _s.sort_values(metric,ascending=False)
                sorted_models = z['Model'].values
                
                for i in range(len(sorted_models)):
                    rel_rank['Model'].append(sorted_models[i])
                    rel_rank['Drug'].append(drug)
                    rel_rank['Metric'].append(metric)
                    rel_rank['Screen'].append(screen)
                    rel_rank['Eval Type'].append(et)
                    rel_rank['Rank'].append(i+1)
            
            for metric in ['NDCG','Perm Overlap']:
                z = _r[_r['K']==10]
                z = z.sort_values(metric,ascending=False)
                sorted_models = z['Model'].values
                for i in range(len(sorted_models)):
                    rel_rank['Model'].append(sorted_models[i])
                    rel_rank['Drug'].append(drug)
                    rel_rank['Metric'].append(metric)
                    rel_rank['Screen'].append(screen)
                    rel_rank['Eval Type'].append(et)
                    rel_rank['Rank'].append(i+1)
            
     


scalar_res, rank_res = pd.concat(scalar_res),pd.concat(rank_res)
rel_rank = pd.DataFrame(rel_rank)

rel_rank = rel_rank[['Model','Metric','Screen','Eval Type','Rank']]
rel_rank =rel_rank.groupby(['Model','Metric','Screen','Eval Type']).mean().reset_index()
rel_rank.to_csv("../results/relative_rank.csv")


for screen in screens:
    sub_dir = f"../figs/fig2/{screen}"
    os.makedirs(sub_dir,exist_ok=True)
    et = "Seen"
    
    plot_models = ['Ground Truth', 'DeepCINET-0','DeepCINET-50','DeepCINET-95']
    
    
    _sr = scalar_res[
        (scalar_res['Screen']==screen) & 
        (scalar_res['Eval Type']==et) & 
        (scalar_res['Model'].isin(plot_models))]
    
    _rr = rank_res[
        (rank_res['Screen']==screen) & 
        (rank_res['Eval Type']==et) & 
        (rank_res['Model'].isin(plot_models))]
    

    for metric in ['Concordance','Pearson']:
        sns.barplot(_sr,y='Drug',x=metric,hue='Model')
        if metric=='Concordance':
            plt.xlim(0.4,1)
            
        plt.title(f"{metric} Performance for {et} CCLs in {screen}")
        plt.legend().remove()
        plt.tight_layout()
        plt.savefig(f"{sub_dir}/{metric}_{et}.png",dpi=500)
        plt.close()

        gt = _sr[_sr['Model']=='Ground Truth']
        gt = {i:j for i,j in zip(gt['Drug'],gt[metric])}

        _st = _sr[_sr['Model']!='Ground Truth']
        drugs = pd.unique(_sr['Drug'])
            # for drug in drugs:

       

        # lefts = np.zeros(n_drugs)
        dataset = []
        data_orders = []
        dnames = []
        drugs = sorted(drugs)
        for drug in drugs:
            # print(drug)
            dnames.append(drug)
            _d = _st[_st['Drug']==drug]
            
            if drug=='Gemcitabine':
                print(_d)
            _d = _d.sort_values(by=[metric])
            
            
            # sys.exit()
            drug_res = {}
            _d.reset_index(inplace=True)
            data_orders.append([x for x in _d['Model']])
            last = 0
            for i,row in _d.iterrows():
                gap = row[metric]-last
                drug_res[row['Model']]=gap
                last = row[metric]                    
                
            
            dataset.append(drug_res)
        

        
        # print(orders)
        colors = sns.color_palette("colorblind", 4)
        # colors = ["r","g","b","y"]
        names = sorted(dataset[0].keys())

        values = np.array([[data[name] for name in order] for data,order in zip(dataset, data_orders)])
        lefts = np.insert(np.cumsum(values, axis=1),0,0, axis=1)[:, :-1]
        orders = np.array(data_orders)
        bottoms = np.arange(len(data_orders))
        # dnames = ["data %d" % (t+1) for t in bottoms]
        for name, color in zip(names, colors):
            idx = np.where(orders == name)
            value = values[idx]
            left = lefts[idx]
            plt.barh(dnames,left=left, width=value, height=0.5,
                    color=color, label=name)
        # plt.legend(loc="best", bbox_to_anchor=(1.0, 1.00))

        for i, d in enumerate(drugs):
                # drug =  tick.get_text()
            ground_truth = gt[d]
            plt.plot(ground_truth, i, "|", markersize=15, color="k")
        plt.subplots_adjust(right=0.85)
        # plt.title(f"{metric}@{k} for {et} CCLs in {screen}")
        plt.title(f"{metric} Performance for {et} CCLs in {screen}")
        # plt.show()
        
        plt.tight_layout()
        plt.savefig(f"{sub_dir}/{metric}_{et}_stacked.png",dpi=500)
        
        plt.legend(loc="best", bbox_to_anchor=(1.0, 1.00))
        plt.tight_layout()
        plt.savefig(f"{sub_dir}/{metric}_{et}_stacked_legend.png",dpi=500)
        plt.close()
        
    
    for metric in ['NDCG','Perm Overlap']:
        for k in [5,10]:
            _r = _rr[_rr['K']==k]
            _t = _r[_r['Model']!='Ground Truth']


            gt = _r[_r['Model']=='Ground Truth']
            gt = {i:j for i,j in zip(gt['Drug'],gt['NDCG'])}
            
            total = _r.groupby('Drug')['NDCG'].sum().reset_index()
            totals = {i:j for i,j in zip(total['Drug'],total['NDCG'])}
            
            # ndgs = []
            # for j, row in _r.iterrows():
            #     ndgs.append(row['NDCG']/totals[row['Drug']])
            
            # _r['NDCG']=ndgs
            
            
                
            

            n_drugs = len(pd.unique(_r['Drug']))
            fig, ax = plt.subplots()
            
            drugs = pd.unique(_r['Drug'])
            # print(drugs)
            # print(len(drugs))
            # sys.exit()
            # for drug in drugs:

            res = {}
            for model in ['DeepCINET-0','DeepCINET-50','DeepCINET-95','Ground Truth']:
                temp = _r[_r['Model']==model]
                # print(temp)
                model_res = []
                for drug in drugs:
                    if temp[temp['Drug']==drug].shape[0]==0:
                        model_res.append(np.nan)
                    else:
                        model_res.append(temp[temp['Drug']==drug][metric].values[0])


                res[model] = np.array(model_res)
            

            lefts = np.zeros(n_drugs)
            dataset = []
            data_orders = []
            dnames = []
            drugs = sorted(drugs)
            for drug in drugs:
                # print(drug)
                dnames.append(drug)
                _d = _t[_t['Drug']==drug]
                _d = _d.sort_values(by=[metric])
                drug_res = {}
                _d.reset_index(inplace=True)
                data_orders.append([x for x in _d['Model']])
                last = 0
                for i,row in _d.iterrows():
                    gap = row[metric]-last
                    drug_res[row['Model']]=gap
                    last = row[metric]                    
                    # if i==0
                
                dataset.append(drug_res)
            

            
            # print(orders)
            
            colors = sns.color_palette("colorblind", 4)
            # colors = ["r","g","b","y"]
            names = sorted(dataset[0].keys())

            values = np.array([[data[name] for name in order] for data,order in zip(dataset, data_orders)])
            lefts = np.insert(np.cumsum(values, axis=1),0,0, axis=1)[:, :-1]
            orders = np.array(data_orders)
            bottoms = np.arange(len(data_orders))
            # dnames = ["data %d" % (t+1) for t in bottoms]
            # ax = plt.plot()
            for name, color in zip(names, colors):
                idx = np.where(orders == name)
                value = values[idx]
                left = lefts[idx]
                plt.barh(dnames,left=left, width=value, height=0.5,
                        color=color, label=name)
                
            
            
            for i, d in enumerate(drugs):
                # drug =  tick.get_text()
                if d in gt.keys():
                    ground_truth = gt[d]
                    plt.plot(ground_truth, i, "|", markersize=15, color="k")
                
            # plt.legend(loc="best", bbox_to_anchor=(1.0, 1.00))
            plt.subplots_adjust(right=0.85)
            plt.title(f"{metric}@{k} for {et} CCLs in {screen}")
            plt.tight_layout()
            plt.savefig(f"{sub_dir}/{metric}__{k}_{et}.png",dpi=500)
            
            # plt.legend(loc="best", bbox_to_anchor=(1.0, 1.00))
            # plt.tight_layout()
            # plt.savefig(f"{sub_dir}/{metric}__{k}_{et}_legend.png",dpi=500)
            plt.close()


scalar_improvements = defaultdict(list)
rank_improvements = defaultdict(list)
for screen in ['gCSI','GDSC2','GDSC1']:

    sub_dir = f"../figs/fig2/{screen}"
    os.makedirs(sub_dir,exist_ok=True)
    et = "Unseen"
    neural_models = ['DeepCINet-0','DeepCINet-50','DeepCINet-95']
    plot_models = ['DeepCINet-0','DeepCINet-95','DeepCINet-50','ElasticNet','RandomForest','Univariable']
    valid_drugs = pd.unique(
            sizes[(sizes['Test Screen']==screen) & (sizes['Num. Seen']>=seen_cutoff) & (sizes['Drug'].isin(drugs))]['Drug']
            )
    
    for drug in valid_drugs:
        # subset neural data
            _ns = neural_scalars[(neural_scalars['Drug']==drug) & 
                                 (neural_scalars['Screen']==screen) &
                                 (neural_scalars['Eval Type']==et)
                                 ][['Model','Drug','Screen','Pearson','Concordance','Eval Type']]
            
            _nr = neural_ranking[(neural_ranking['Drug']==drug) & 
                                 (neural_ranking['Screen']==screen) &
                                 (neural_ranking['Eval Type']==et)
                                 ][['Model','Drug','Screen','K','NDCG','Perm Overlap', 'Eval Type']]
            
            # subset baselines

            _bs = baseline_scalars[(baseline_scalars['Drug']==drug) & 
                                 (baseline_scalars['Screen']==screen) &
                                 (baseline_scalars['Eval Type']==et)
                                 ][['Model','Drug','Screen','Pearson','Concordance','Eval Type']]
            
            
            _br = baseline_ranking[(baseline_ranking['Drug']==drug) & 
                                 (baseline_ranking['Screen']==screen) &
                                 (baseline_ranking['Eval Type']==et)
                                 ][['Model','Drug','Screen','K','NDCG','Perm Overlap', 'Eval Type']]
            
            for metric in ['Concordance','Pearson']:
                best_perf = np.max(_bs[metric].values)
                best = _bs[_bs[metric]==best_perf]
                
                scalar_improvements['Drug'].append(drug)
                scalar_improvements['Screen'].append(screen)
                scalar_improvements['Metric'].append(metric)
                scalar_improvements['Model'].append('Best Baseline')
                scalar_improvements['Performance'].append(best_perf)
                scalar_improvements['Gap Vs. Baseline'].append(best_perf-best_perf)
                
                for model in neural_models:
                    print(_ns)
                    
                    temp = _ns[_ns['Model']==model]
                    print(temp)
                    print(scalar_improvements)
                    scalar_improvements['Drug'].append(drug)
                    scalar_improvements['Screen'].append(screen)
                    scalar_improvements['Metric'].append(metric)
                    scalar_improvements['Model'].append(model)
                    scalar_improvements['Performance'].append(temp[metric].values[0])
                    scalar_improvements['Gap Vs. Baseline'].append(temp[metric].values[0]-best_perf)
            
scalar_improvements  = pd.DataFrame(scalar_improvements)              
for screen in ['gCSI','GDSC2','GDSC1']:
    
    for metric in ['Pearson','Concordance']:
        temp = scalar_improvements[
            (scalar_improvements['Screen']==screen) & (scalar_improvements['Metric']==metric)
        ]

        sns.barplot(temp, x= 'Performance',y='Drug',hue='Model')
        plt.title(f"Performance Gap in {metric} in {screen}")
        if metric=='Concordance':
            plt.xlim(0.4,1)
        plt.tight_layout()
        plt.show()
    
            
        
        
        

# # _r, _b = [], []
# # krange = np.concatenate( ([1], np.arange(5,25,5)))
# # print(krange)   deltas = list(pd.unique(res['Percentile']))






    
    
    
    


# #         sns.barplot(results[results['Screen']==screen],y='Drug',x=metric,hue='Model')
# #         if metric=='Concordance':
# #             plt.xlim(0.4,1)

# #         plt.tight_layout()
# #         plt.savefig(f"{fig_dir}/{screen}_{metric}_barplot.png")
# #         plt.close()
# #         sns.barplot(ml[ml['Screen']==screen],y='Drug',x=metric,hue='Model')
# #         if metric=='Concordance':
# #             plt.xlim(0.4,1)
# #         plt.title(f"{screen} {metric} on Seen Cell Lines")
# #         plt.tight_layout()
# #         plt.savefig(f"{fig_dir}/{screen}_{metric}_baselines_barplot.png")
# #         plt.close()




# # sys.exit()
# # res = pd.read_csv("../results/neurals/scalars.csv")

# # improvements = defaultdict(list)
# # for model in ['ECINET', 'One Layer', 'Two Layer']:
# #     for drug in drugs:
# #         _d = res[(res['Drug']==drug) & (res['Model']==model)]

# #         for et in ['Seen','Unseen']:
# #             for metric in ['Pearson','Concordance']:
# #                 for dd in deltas[1:]:

# #                     zero_score = _d[(_d['Percentile']==0) & (_d['Eval Type']==et)][metric].values[0]
# #                     delta_score= _d[(_d['Percentile']==dd)  & (_d['Eval Type']==et)][metric].values[0]
# #                     improvements['Drug'].append(drug)
# #                     improvements['Model'].append(model)
# #                     improvements['Metric'].append(metric)
# #                     improvements['Percentile'].append(dd)
# #                     improvements['Eval Type'].append(et)
# #                     improvements['Zero Score'].append(zero_score)
# #                     improvements['Delta Score'].append(delta_score)
# #                     improvements['Gap'].append(delta_score-zero_score)
# #                     improvements['Substantial'].append(delta_score>=1.03*zero_score)



# # im = pd.DataFrame(improvements)
# # # im = im[im['Drug'].isin(drugs)]
# # # im = im[im['Substantial']==True]
# # # im['Percentile'] = [str(x) for x in im['Percentile']]
# # tl = im[im['Model']=='Two Layer']
# # # tl = tl[tl['Percentile']==95]
# # z = tl[tl['Metric']=='Pearson']
# # z = z[z['Percentile'].isin([0,95])]
# # z = z[z['Eval Type']=='Unseen']
# # z['Percentile']=[str(x) for x in z['Percentile']]

# # tl = tl[tl['Eval Type']=='Seen']

# # tl = tl[tl['Metric']=='Pearson']
# # labs = np.round(tl['Delta Score'].values,2)
# # tl['Positive Delta Performance'] = pd.cut(
# #     tl['Delta Score'],
# #     bins=[-1,0.2,0.3,0.5,1],
# #     labels = ["<0.2","0.2-0.3","0.3-0.5",">0.5"])
# # palette = {'<0.2': "#919191", '0.2-0.3':  "#CF8B8B",'0.3-0.5':  "#E46262", '>0.5': "#FF0000"}
# # ax = sns.barplot(tl[tl['Metric']=='Pearson'], x='Gap', y='Drug',hue='Positive Delta Performance',palette=palette)
# # # ax.bar_label(ax.containers[-1], labels=labs,fontsize=10)
# # ax.set(xlim=(-0.6,0.4))
# # plt.tight_layout()
# # # plt.show()
# # plt.savefig(f"{fig_dir}/pearson_gap.png")
# # plt.close()
# # # ax.bar_label(ax.containers[-1], fmt='Mean:\n%.2f', label_type='center')
# # # plt.tight_layout()

# # # plt.close()
# # # sns.catplot(tl[tl['Metric']=='Concordance'], x='Gap', y='Drug', col = 'Eval Type',kind = 'bar')
# # # plt.tight_layout()
# # # plt.savefig("../figs/two_layer_ci.png")
# # # plt.close()

# # # # im = im[im['Percentile']==95]
# # # # im = im[im['Model'].isin(['ECINET','Two Layer'])]
# # # # sns.catplot(im[im['Metric']=='Pearson'], x='Gap', y='Drug', col = 'Eval Type',kind = 'bar',hue='Model')
# # # # plt.tight_layout()

# # # # plt.savefig("../figs/all_pearson.png")
# # # # plt.close()
# # # # sns.catplot(im[im['Metric']=='Concordance'], x='Gap', y='Drug', col = 'Eval Type',kind = 'bar',hue= 'Model')
# # # # plt.tight_layout()
# # # # plt.savefig("../figs/all_ci.png")
# # # # plt.close()

# # # # for screen in ['Pooled']:
# # # # 	_im = im[(im['Screen']==screen)]
# # # # 	sns.catplot(_im,kind = 'bar', x = "Gap", y="Drug",row = "Eval Type", col = "Metric",hue='Percentile')
# # # # 	plt.tight_layout()
# # # # 	plt.savefig("../figs/cat_growth.png",dpi=500)
# # # # 	# plt.show()
# # # # 	plt.close()
# # # # 	_temp = df[df['Screen']==screen]
# # # # 	_temp['Delta_bin'] = ["Positive" if float(x)>0 else "Zero" for x in _temp['Percentile']]
# # # # 	for metric in ['Pearson','Concordance']:
# # # # 		plt.title(f"{screen} - {metric}")
# # # # 		sns.violinplot(_temp, x= 'Eval Type',y=metric,hue= "Delta_bin")
# # # # 		plt.show()
# # # # 	sys.exit()
