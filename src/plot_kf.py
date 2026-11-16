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

drugs = utils.read_text_file("../procdata/drug_lists/common_drugs.txt")
screens = ['gCSI','GDSC2']
results = defaultdict(list)
ranking_results = defaultdict(list)


kr = [1,5,10,15,20]

for screen in screens:
    if screen=='Pooled':
        kmax = 20
    else:
        kmax = 15


    for drug in tqdm.tqdm(drugs,leave=False):  
        for model in ['ECINET','ONE_LAYER','TWO_LAYER']:
            if screen == 'Pooled':
                s1  = pd.read_csv(f"../results/{drug}/neurals/kfold/test/gCSI/{model}.csv",index_col=0),
                s2  = pd.read_csv(f"../results/{drug}/neurals/kfold/test/GDSC2/{model}.csv",index_col=0)
                neural = pd.concat([s1,s2])
            else:
                neural = pd.read_csv(f"../results/{drug}/neurals/kfold/test/{screen}/{model}.csv",index_col=0)	
            train = pd.read_csv(f"../results/{drug}/neurals/kfold/train/{model}.csv",index_col=0)
            seen_cells = pd.unique(train['Cell Line'])
            for crit in ['ci','pearson']:
                _temp = neural[neural['Criteria']==crit]
                seen = _temp[_temp['Cell Line'].isin(seen_cells)]
                unseen = _temp[~_temp['Cell Line'].isin(seen_cells)]
                
                for et, _df in zip(['Seen','Unseen'],[seen,unseen]):
                    _km = min(_df.shape[0],kmax)
                    krange = [x for x in kr if x<_km]
                    corr,_ = pearsonr(_df['Predicted'],_df['AAC'])
                    ci = evm.robust_concordance(_df['Predicted'].values,_df['AAC'].values)
                    delta = _df['Delta'].values[0]
                    results['Drug'].append(drug)
                    results['Model'].append(model)
                    results['Screen'].append(screen)
                    results['Criteria'].append(crit)
                    results['Eval Type'].append(et)
                    results['Delta'].append(delta)
                    results['Pearson'].append(corr)
                    results['Concordance'].append(ci)

                    for K in krange:
                    
                        po = evm.perm_overlap(_df['AAC'].values,_df['Predicted'].values,K)
                        ndcg = evm.ndcg_at_k(_df['AAC'].values,_df['Predicted'].values,K)
                        ranking_results['Model'].append(model)
                        ranking_results['Drug'].append(drug)
                        ranking_results['Screen'].append(screen)
                        # baseline_results['Percentile'].append(p)
                        ranking_results['K'].append(K)
                        # baseline_results['Delta'].append(deltas[p])
                        
                        ranking_results['Eval Type'].append(et)
                        ranking_results['NDCG'].append(ndcg)
                        ranking_results['Perm Overlap'].append(po)
                


results = pd.DataFrame(results)
results.to_csv("../results/neurals/kfold_scalars.csv")
ranking_results = pd.DataFrame(ranking_results)
ranking_results.to_csv("../results/neurals/kfold_ranking.csv")