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

# d = pd.read_csv("../procdata/Broad/Erlotinib/data.csv")
# print(d.head())

# print(d.shape)

# sys.exit()
drugs = utils.read_text_file("../procdata/drug_lists/post_qc.txt")
screens = ['GDSC2','gCSI','GDSC1']
results = defaultdict(list)
eval_type = 'Unseen'

bl = pd.read_csv("../results/baseline_scalars.csv",index_col=0)
n = pd.read_csv("../results/neurals/kfold_scalars.csv",index_col=0)
cl = pd.read_csv("../results/drug_classification.csv")

drug_to_class = {k:v for k,v in zip(cl['Drug'].values,cl['Class'].values)}



print(drug_to_class)
# print(cl)

n = n[n['Model'].isin(['TWO_LAYER'])]
n['Model']= 'DeepCINet-CV'
rg = pd.read_csv("../results/neurals/scalars.csv",index_col=0)
rg['Model'] = ["DeepCINet-"+str(x) for x in rg['Percentile']]



# improvements = defaultdict(list)
bvb= defaultdict(list)
full_res = []
for screen in screens:
    for drug in drugs:
        for eval_type in ['Seen','Unseen']:
            _n = n[(n['Drug']==drug) & (n['Screen']==screen) & (n['Criteria']=='pearson')]
            _g = rg[(rg['Drug']==drug) & (rg['Screen']==screen) &(rg['Eval Type']==eval_type) & (rg['Percentile'].isin([0,50,95]))]
            _g['Model'] = ["DeepCINet-"+str(x) for x in _g['Percentile']]
            _b = bl[(bl['Drug']==drug) &
                    (bl['Screen']==screen) &
                    (bl['Eval Type']==eval_type)]
            
            _n = _n[_n['Eval Type']==eval_type]
            fr = pd.concat([_n,_g,_b])
            fr = fr[['Drug', 'Model','Screen','Eval Type', 'Pearson','Concordance']]
            fr['Drug Class'] = drug_to_class[drug]
            full_res.append(fr)
            _n = pd.concat([_n,_g])

            
            for metric in ['Pearson','Concordance']:
                for model, df in zip(['Best Neural','Best Baseline'],[_n,_b]):
                    best_perf = np.max(df[metric])
                    # best_neural = np.max(_n[metric])
                    
                    best_model = df[df[metric]==best_perf]['Model'].values[0]
                    
                   
                    

                    bvb['Drug'].append(drug)
                    bvb['Screen'].append(screen)
                    bvb['Metric'].append(metric)
                    bvb['Model'].append(model)
                    bvb['Best Model'].append(best_model)
                    bvb['Eval Type'].append(eval_type)
                    bvb['Performance'].append(best_perf)
                
                    
# print(pd.DataFrame(bvb))

full_res = pd.concat(full_res)
# print(full_res)
# print(full_res.head())


# for screen in ['GDSC2','gCSI']:
#     for et in ['Unseen','Seen']:
#         _t = full_res[full_res['Screen']==screen]
#         _t = _t[_t['Eval Type']==et]
#         _t = _t[_t['Model'].isin(['DeepCINet-CV','DeepCINet-0','DeepCINet-50','DeepCINet-95'])]
#         # print(_t.head())
#         # sys.exit()
#         sns.violinplot(_t, y='Pearson',x='Drug Class')
#         plt.title(f"In {screen} {et}")
#         plt.show()


bvb = pd.DataFrame(bvb)
print(bvb)




for z in ['Best Neural', 'Best Baseline']:
    for p in ['Seen','Unseen']:
        _b = bvb[bvb['Model']==z]
        _b = _b[_b['Eval Type']==p]
        _b = _b[_b['Metric']=='Pearson']
        

        for screen in ['gCSI','GDSC2']:
            temp = _b[_b['Screen']==screen]
            print(screen)
            print(temp)
# print(temp['Best Model'].value_counts())
# sys.exit()    

diffs = defaultdict(list)

for screen in ['gCSI','GDSC2','GDSC1']:
    for metric in ['Pearson']:
        for et in ['Seen', 'Unseen']:
            _temp = bvb[bvb['Screen']==screen]
            _temp = _temp[_temp['Metric']==metric]
            _temp = _temp[_temp['Eval Type']==et]
            for drug in pd.unique(_temp['Drug']):
                _d = _temp[_temp['Drug']==drug]
                
                neur = _d[_d['Model']=='Best Neural']['Performance'].values[0]
                
                base = _d[_d['Model']=='Best Baseline']['Performance'].values[0]
                diffs['Drug'].append(drug)
                diffs['Screen'].append(screen)
                diffs['Eval Type'].append(et)
                diffs['Metric'].append(metric)
                diffs['Gap'].append(neur-base)
                diffs['Signif'].append(neur>1.15*base)
                diffs['Neural'].append(neur)
                diffs['Baseline'].append(base)
                diffs['Max'].append(max(neur,base))

                



                
                
                
            
            sns.barplot(_temp,x="Performance",y="Drug",hue="Model")
            plt.title(f"{metric} performance on {et} Cell Lines in {screen}")
            if metric == 'Concordance':
                plt.xlim(0.4,1)
            plt.tight_layout()
            plt.savefig(f"../figs/{screen}_{metric}_{et}.png",dpi=500)
            # plt.show()
            plt.close()


diffs = pd.DataFrame(diffs)
print(diffs[
    (
        diffs['Signif']==True
        ) 
        & (
        diffs['Eval Type']=='Unseen')
        ]
)

values = defaultdict(list)

for drug in pd.unique(diffs['Drug']):
    Broad = pd.read_csv(f"../procdata/Broad/{drug}/data.csv",usecols=['Cell Line','AAC'])
    values['Drug'].append(drug)
    values['IQR'].append( np.quantile(Broad['AAC'],.75) -  np.quantile(Broad['AAC'],.25))
    values['Mean'].append(np.mean(Broad['AAC']))
    values['Skewness'].append(skew(Broad['AAC']))
    x,p = skewtest(Broad['AAC'])
    values['p value'].append(p)
    

    values['Kurtosis'].append(kurtosis(Broad['AAC']))


values = pd.DataFrame(values)

values = values[['Drug','IQR', 'Kurtosis', 'Skewness','p value']].drop_duplicates()
# print(np.mean(_v['Kurtosis']))
# print(np.mean(valu['Skewness']))
values.to_csv("../results/skewness.csv")



first = []
last = []
for i, r in cl.iterrows():
    if r['Class']=='Narrow Effect':
        first.append(r['Drug'])
    else:
        last.append(r['Drug'])



ordered_drugs = first + last
print(first)
print(last)
order_map = {ordered_drugs[i]:i for i in range(len(ordered_drugs))}
sizes = pd.read_csv("../results/dataset_sizes.csv")
print(sizes)
# sys.exit()
for z in pd.unique(diffs['Eval Type']):
    for p in pd.unique(diffs['Screen']):
        d = diffs[diffs['Eval Type']==z]
        d = d[d['Screen']==p]
        
        size_temp = sizes[sizes['Test Screen']==p][['Drug',f"Num. {z}"]]
        d2s = {d:s for d,s in zip(size_temp['Drug'],size_temp[f"Num. {z}"])}
        d2g = {d:g for d,g in zip(d['Drug'],d['Gap'])}

        # d = d.sort_values(by=['Drug'], key = lambda x: x.map(order_map))
        

        d = d.sort_values(by=['Gap'])
        
        com = d[['Drug','Gap']].merge(values,on='Drug')
        print(p)
        print(z)
        print(com[['Gap','IQR','Kurtosis','Skewness']].corr())
        # sys.exit()
        
        d['Baseline Performance'] = pd.cut(
            d['Baseline'], 
            bins=[-1,0.2,0.3,0.5,1],
            labels = ["<0.2","0.2-0.3","0.3-0.5",">0.5"])
        palette = {'<0.2': "#919191", '0.2-0.3':  "#CF8B8B",'0.3-0.5':  "#E46262", '>0.5': "#FF0000"}
        
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
            
            ax.text(loc, tick.get_position()[1], s=f"{d2s[drug]}")
        # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.xlim(x0, x1+offset)
        plt.title(f"Best Neural vs. Best Baseline Performance Gap\nin {z} Cell Lines in {p}")
        plt.tight_layout()
        # plt.show()
        # sys.exit()
        plt.savefig(f"../figs/kfold_gaps_{z}_{p}.png",dpi=500)
        plt.close()

        # if z == 'Unseen':
        #     res = d[d['Signif']==True]
        #     res =  res[['Drug','Gap']].merge(values,on='Drug')
        #     # print(res)
            
        #     print(pearsonr(res['Gap'].values,res['Skewness'].values))
        #     print(pearsonr(res['Gap'].values,res['Kurtosis'].values))
        #     print(pearsonr(res['Gap'].values,res['IQR'].values))
        #     print("\n")

