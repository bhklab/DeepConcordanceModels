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


neural_to_model = {'ECINET':"ECINET",'ONE_NARROW':"One Layer", "TWO_NARROW":"Two Layer"}


drugs = utils.read_text_file("../procdata/drug_lists/post_qc.txt")




screens = ['gCSI','GDSC2','GDSC1']

ep=30



# print(f"Working on {exp_type}")
neural_results = defaultdict(list)

all_seen, all_unseen = [],[]
for screen in screens:
    for drug in tqdm.tqdm(drugs,leave=False):  
        for model in ['TWO_NARROW']:
            if screen == 'Pooled':
                s1  = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/test/GDSC2/{model}.csv",index_col=0)
                s2  = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/test/gCSI/{model}.csv",index_col=0)
                neural = pd.concat([s1,s2])
            else:
                neural = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/test/{screen}/{model}.csv",index_col=0)	
            train = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/train/{model}.csv",index_col=0)
            

            deltas = {}
            for k in pd.unique(neural['Percentile']):
                if k in [0,50,75,95]:
                    _t = neural[neural['Percentile']==k]
                    deltas[str(k)] = _t['Delta'].values[0]
            
            # deltas = {k:v f}
            
        

        
            
            neural = neural[neural['Epoch'].isin([ep])]
            train = train[train['Epoch'].isin([ep])]
            seen_cells = pd.unique(train['Cell Line'])
            
            
            
            for p in deltas:
                
                _temp = neural[neural['Percentile']==float(p)]
                seen = _temp[_temp['Cell Line'].isin(seen_cells)]
                unseen = _temp[~_temp['Cell Line'].isin(seen_cells)]

                seen['Drug']=drug
                seen['Delta'] = p
                unseen['Drug']=drug
                unseen['Delta']=p
                all_seen.append(seen)
                all_unseen.append(unseen)

                
                
                
                for et, _df in zip(['Seen','Unseen'],[seen,unseen]):
                    
                    sns.scatterplot(_df,y='Predicted',x='AAC')
                    plt.title(f"DeepCINet Predictions vs AAC\n{drug} {et} Cell lines\n Delta = {p}")
                    plt.savefig(f"../figs/{drug}/solid/{et}_{model}__{p}_pred_vs_aac.png")
                    plt.close()
                   
                    
                
                    
                    corr,_ = pearsonr(_df['Predicted'],_df['AAC'])
                    ci = evm.robust_concordance(_df['Predicted'].values,_df['AAC'].values)
                    rci = evm.robust_concordance(_df['Predicted'].values,_df['AAC'].values,delta_o=deltas[p])
                   
                    # if p!="0":
                    #     print(p)
                    #     print(ci)
                    #     print(rci)
                        


                    neural_results['Model'].append(neural_to_model[model])
                    neural_results['Drug'].append(drug)
                    neural_results['Screen'].append(screen)
                    neural_results['Percentile'].append(p)
                    
                    neural_results['Delta'].append(deltas[p])
                    # neural_results['Epoch'].append(e)
                    neural_results['Eval Type'].append(et)
                    neural_results['Pearson'].append(corr)
                    neural_results['Concordance'].append(ci)
                    neural_results['Robust Concordance'].append(rci)


all_seen = pd.concat(all_seen)

neural_results = pd.DataFrame(neural_results)
res_dir = f"../results/neurals/"
os.makedirs(res_dir,exist_ok=True)
neural_results.to_csv(f"{res_dir}/scalars.csv")