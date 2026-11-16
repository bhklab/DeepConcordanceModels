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


n = pd.read_csv("../results/neurals/kfold_scalars.csv",index_col=0)
b = pd.read_csv("../results/baseline_scalars.csv",index_col=0)

n =n[n['Criteria']=='pearson']
drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")
results = defaultdict(list)
stacked_res = []

for drug in drugs:
    _b = b[b['Drug']==drug]
    

    _n = n[n['Drug']==drug]
    
    

    bb = []
    for et in ['Seen','Unseen']:
        
        for metric in ['Pearson','Concordance']:
            best_b = _b[_b[metric]==np.max(_b[_b['Eval Type']==et][metric])]
            bb.append(best_b)
            model_b = best_b['Model'].values[0]
            
            best_n = _n[_n[metric]==np.max(_n[_n['Eval Type']==et][metric])]
            
            for model in ['ECINET','ONE_LAYER','TWO_LAYER','BEST_NEURAL']:
                if model!='BEST_NEURAL':

                    _nt = _n[(_n['Model']==model) & (_n['Eval Type']==et)]
                    
                   
                    gap = _nt[metric].values[0]-best_b[metric].values[0]
                    
                    
                    results['Drug'].append(drug)
                    results['Metric'].append(metric)
                    results['Eval Type'].append(et)
                    results['Baseline Model'].append(best_b['Model'].values[0])
                    results['Baseline Performance'].append(best_b[metric].values[0])
                    results['Neural Model'].append(_nt['Model'].values[0])
                    results['Neural Performance'].append(_nt[metric].values[0])
                    results['Gap'].append(gap)
                    
                else:
                    gap = best_n[metric].values[0]-best_b[metric].values[0]
                    results['Drug'].append(drug)
                    results['Metric'].append(metric)
                    results['Eval Type'].append(et)
                    results['Baseline Model'].append(best_b['Model'].values[0])
                    results['Baseline Performance'].append(best_b[metric].values[0])
                    results['Neural Model'].append("Best")
                    results['Neural Performance'].append(best_n[metric].values[0])
                    results['Gap'].append(gap)
    bb = pd.concat(bb)
    bb['Model'] = 'Best Baseline'
    stack = pd.concat([_n,bb])
    stack = stack[['Drug','Model','Eval Type', 'Pearson', 'Concordance']]
    
    stacked_res.append(stack)
    
    
            
# stack = pd.concat(stacked_res)
# stack = stack.drop_duplicates()
# print(stack)
# stack = stack[stack['Model']=='TWO_LAYER']
# sns.catplot(stack, x= 'Concordance',y='Drug', kind='bar', hue='Model',col = 'Eval Type')
# plt.show()
# sys.exit()
# #  sns.catplot(stack,x='Gap',y='Drug',hue='Neural Model',row = 'Eval Type',col='Metric',kind='bar')

# sys.exit()
res = pd.DataFrame(results)
# print(res.head())
# sys.exit()
# _r = res[res['Eval Type']=='Unseen']
# _r =_r[_r['Metric']=='Pearson']
_r = res[res['Neural Model']=='Best']
sns.catplot(_r,x='Gap',y='Drug',hue='Neural Model',row = 'Eval Type',col='Metric',kind='bar')
plt.tight_layout()
plt.show()


