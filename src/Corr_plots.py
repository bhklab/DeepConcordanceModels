###
#
# Correlation plots
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
from itertools import product 

neural_to_model = {'ECINET':"ECINET",'ONE_NARROW':"One Layer", "TWO_NARROW":"Two Layer"}
drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")
results = defaultdict(list)

baselines = ['RandomForest','ElasticNet','Univariable','KRR']


corres = defaultdict(list)
for screen in ['gCSI', 'GDSC2']:  
    for drug in drugs:  
        model_to_df = {}
        n = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/test/{screen}/TWO_NARROW.csv",index_col=0)
        n = n[n['Epoch']==30]
        for pct in [0,50,95]:
            _t = n[n['Percentile']==pct][['Cell Line','Predicted']]
            _t.columns = ['Cell Line',"Predicted"]
            model_to_df[f"DeepCINet-{pct}"] = _t.copy()

        
        for bm in baselines:
            df = pd.read_csv(f"../results/{drug}/baselines/test/{screen}/{bm}.csv",index_col=0)
            df = df[['Cell Line','Predicted']]
            model_to_df[bm] = df.copy()


        cv_df= pd.read_csv(f"../results/{drug}/neurals/kfold/test/{screen}/TWO_LAYER.csv",index_col=0)
        
        cv_df = cv_df[cv_df['Criteria']=='pearson'][['Cell Line','Predicted']]
        
        model_to_df['DeepCINet-CV'] =cv_df

        for m1, m2 in product(model_to_df.keys(),model_to_df.keys()):
            d1 = model_to_df[m1]
            d2 = model_to_df[m2]
            df = d1.merge(d2,on='Cell Line')
            
            corr,_ = pearsonr(df['Predicted_x'].values,
                              df['Predicted_y'].values)
            

            corres['Drug'].append(drug)
            corres['Model 1'].append(m1)
            corres['Model 2'].append(m2)
            corres['Screen'].append(screen)
            corres['Pearson'].append(corr)
            
corres = pd.DataFrame(corres)

for screen in ['gCSI','GDSC2']:
    _c = corres[corres['Screen']==screen]
    _c =_c[['Model 1' ,'Model 2', 'Pearson']].groupby(['Model 1','Model 2']).mean().reset_index()
    # print(_c)
    rp = _c.pivot(index='Model 1', columns='Model 2',values='Pearson')
    ax = sns.heatmap(rp,annot=True, fmt=".3f")
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    plt.xticks(fontsize=7,rotation=30,ha='left')
    plt.title(f"Correlation Between Model Predictions in {screen}")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"../figs/Pearson_{screen}_heatmap.png",dpi=500)
    plt.close()
# sys.exit()                
                
                
#                
#                 results['Drug'].append(drug)
#                 results['Model'].append(model)
#                 results['Percentile'].append(pct)
#                 results['Baseline'].append(bm)
#                 results['Correlation'].append(corr)


# results = pd.DataFrame(results)
# print(results)
# # sys.exit()
# results = results[results['Model']=='TWO_NARROW']
# for drug in drugs:
#     t = results[results['Drug']==drug]
#     sns.barplot(t,x='Percentile',y='Correlation',hue='Baseline')
#     plt.title(f"{drug}")
#     plt.show()    