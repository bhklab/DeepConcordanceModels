from scipy.stats import skew, kurtosis, skewtest
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import utils
import numpy as np 
from collections import defaultdict
import os
import sys
from scipy.stats import pearsonr
import evaluation_metrics as evm
import tqdm
from collections import defaultdict

drugs = utils.read_text_file("../procdata/drug_lists/post_qc.txt")
screens = ['GDSC2','gCSI','GDSC1']

bl = pd.read_csv("../results/baseline_ranking.csv",index_col=0)
n = pd.read_csv("../results/neurals/kfold_ranking.csv",index_col=0)
cl = pd.read_csv("../results/drug_classification.csv")

drug_to_class = {k:v for k,v in zip(cl['Drug'].values,cl['Class'].values)}

# print(cl)
# sys.exit()
n = n[n['Model'].isin(['TWO_LAYER'])]
n['Model']= 'DeepCINet-CV'
rg = pd.read_csv("../results/neurals/ranking.csv",index_col=0)
rg['Model'] = ["DeepCINet-"+str(x) for x in rg['Percentile']]


# improvements = defaultdict(list)
bvb= defaultdict(list)
full_res = []
for screen in screens:
    for drug in drugs:
        for eval_type in ['Seen','Unseen']:
            _n = n[(n['Drug']==drug) & (n['Screen']==screen)]
            
            _g = rg[(rg['Drug']==drug) & (rg['Screen']==screen) &(rg['Eval Type']==eval_type) & (rg['Percentile'].isin([0,50,95]))]
            _g['Model'] = ["DeepCINet-"+str(x) for x in _g['Percentile']]
            _b = bl[(bl['Drug']==drug) &
                    (bl['Screen']==screen) &
                    (bl['Eval Type']==eval_type)]
            
            

            
            _n = _n[_n['Eval Type']==eval_type]
            
            fr = pd.concat([_n,_g,_b])
            
            fr = fr[['Drug', 'Model','Screen','Eval Type', 'K','NDCG','Perm Overlap']]
            
            fr['Drug Class'] = drug_to_class[drug]
            full_res.append(fr)
            _n = pd.concat([_n,_g])

            
            for metric in ['NDCG','Perm Overlap']:
                for k in [5,10]:
                    for model, df in zip(['Best Neural','Best Baseline'],[_n,_b]):
                        _temp = df[df['K']==k]
                        best_perf = np.max(_temp[metric])
                        # best_neural = np.max(_n[metric])
                        if _temp.shape[0]!=0:
                            best_model = _temp[_temp[metric]==best_perf]['Model'].values[0]
                            
                        
                            

                            bvb['Drug'].append(drug)
                            bvb['Screen'].append(screen)
                            bvb['Metric'].append(metric)
                            bvb['Model'].append(model)
                            bvb['K'].append(k)
                            bvb['Best Model'].append(best_model)
                            bvb['Eval Type'].append(eval_type)
                            bvb['Performance'].append(best_perf)
                        
                        
    # print(pd.DataFrame(bvb))

full_res = pd.concat(full_res)
print(full_res)


bvb = pd.DataFrame(bvb)
print(bvb)


diffs = defaultdict(list)

for screen in ['gCSI','GDSC2','GDSC1']:
    for metric in ['NDCG','Perm Overlap']:
        for k in [5,10]:
            for et in ['Seen', 'Unseen']:
                _temp = bvb[bvb['Screen']==screen]
                _temp = _temp[_temp['Metric']==metric]
                _temp = _temp[_temp['Eval Type']==et]
                _temp = _temp[_temp['K']==k]
                
                for drug in pd.unique(_temp['Drug']):
                    _d = _temp[_temp['Drug']==drug]
                    if _d.shape[0]>1:
                        neur = _d[_d['Model']=='Best Neural']['Performance'].values[0]
                        base = _d[_d['Model']=='Best Baseline']['Performance'].values[0]
                        diffs['Drug'].append(drug)
                        diffs['Screen'].append(screen)
                        diffs['Eval Type'].append(et)
                        diffs['Metric'].append(metric)
                        diffs['K'].append(k)
                        diffs['Gap'].append(neur-base)
                        diffs['Signif'].append(neur>1.15*base)
                        diffs['Neural'].append(neur)
                        diffs['Baseline'].append(base)
                        diffs['Max'].append(max(neur,base))

diffs = pd.DataFrame(diffs)
for z in pd.unique(diffs['Eval Type']):
    for p in pd.unique(diffs['Screen']):
        for k in [5,10]:
            for metric in ['NDCG','Perm Overlap']:
                d = diffs[diffs['Eval Type']==z]
                d = d[d['Screen']==p]
                d = d[d['K']==k]
                d = d[d['Metric']==metric]

                # d2s = {d:s for d,s in zip(size_temp['Drug'],size_temp[f"Num. {z}"])}
                d2g = {d:g for d,g in zip(d['Drug'],d['Gap'])}
                d = d.sort_values(by=['Gap'])
                        
                # com = d[['Drug','Gap']].merge(values,on='Drug')
                # print(p)
                # print(z)
                # print(com[['Gap','IQR','Kurtosis','Skewness']].corr())
                # # sys.exit()
                
                d['Baseline Performance'] = pd.cut(
                    d['Baseline'], 
                    bins=[-1,0.2,0.5,0.75,1],
                    labels = ["<0.2","0.2-0.5","0.5-0.75",">0.75"])
                palette = {'<0.2': "#919191", '0.2-0.5':  "#CF8B8B",'0.5-0.75':  "#E46262", '>0.75': "#FF0000"}
                
                ax = sns.barplot(d, x='Gap', y='Drug',hue='Baseline Performance',palette=palette)
                x0,x1 = ax.get_xlim()
                # sys.exit()
                offset = 0.01
                for tick in ax.get_yticklabels():
                    drug =  tick.get_text()
                    # ground_truth = gt[drug]
        
                    
        
                    if d2g[drug]<0:
                        
                        loc = offset
                    else:
                        loc = max(d2g[drug]+0.001,offset)
                            # loc = -0.02
                        # else:
                        #     loc = -0.01
                        # # loc = -offset
                    
                    # ax.text(loc, tick.get_position()[1], s=f"{d2s[drug]}")
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                # plt.xlim(x0, x1+offset)
                # plt.title(f"Best Neural vs. Best Baseline Performance Gap\nin {z} Cell Lines in {p}")
                # plt.tight_layout()
                # # plt.show()
                # # sys.exit()
                # plt.savefig(f"../figs/kfold_gaps_{z}_{p}.png",dpi=500)
                # plt.close()
                
                # d['Baseline Performance'] = pd.cut(
                #     d['Baseline'], 
                #     bins=[-1,0.2,0.3,0.5,1],
                #     labels = ["<0.2","0.2-0.3","0.3-0.5",">0.5"])
                # palette = {'<0.2': "#919191", '0.2-0.3':  "#CF8B8B",'0.3-0.5':  "#E46262", '>0.5': "#FF0000"}
                # ax = sns.barplot(d, x='Gap', y='Drug')
                plt.title(f"Best Neural vs. Best Baseline {metric}@{k} Gap\nin {z} Cell Lines in {p}")
                plt.tight_layout()
                plt.savefig(f"../figs/{p}_{metric}_{k}_{z}_gaps.png",dpi=500)
                plt.close()