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
drugs = utils.read_text_file("../procdata/drug_lists/post_qc.txt")
screens = ['gCSI','GDSC2','GDSC1']
# screens = ['GDSC1']


scalar_results, ranking_results = defaultdict(list), defaultdict(list)


# drugs = ['Lapatinib']

kr = [1,5,10,15,20]
for screen in screens:
    if screen=='Pooled':
        kmax = 20
    else:
        kmax = 15
  
   
    for drug in (pbar:=tqdm.tqdm(drugs)):  
        pbar.set_description(f"Working on {drug} in {screen}")
        for model in ['KRR','ElasticNet','RandomForest','Univariable']:        
            if screen == 'Pooled':
                s1  = pd.read_csv(f"../results/{drug}/baselines/test/GDSC2/{model}.csv",index_col=0)
                s2  = pd.read_csv(f"../results/{drug}/baselines/test/gCSI/{model}.csv",index_col=0)
                res = pd.concat([s1,s2])
            else:
                res = pd.read_csv(f"../results/{drug}/baselines/test/{screen}/{model}.csv",index_col=0)	

            train = pd.read_csv(f"../results/{drug}/baselines/train/{model}.csv",index_col=0)
            

            
            seen_cells = pd.unique(train['Cell Line'])
            seen = res[res['Cell Line'].isin(seen_cells)]
            unseen = res[~res['Cell Line'].isin(seen_cells)]
            
            for et, _df in zip(['Seen','Unseen'],[seen,unseen]):
                _km = min(_df.shape[0],kmax)
                
                krange = [x for x in kr if x<_km]
                
                corr,_ = pearsonr(_df['Predicted'],_df['AAC'])
                ci = evm.robust_concordance(_df['Predicted'].values,_df['AAC'].values)

                scalar_results['Drug'].append(drug)
                scalar_results['Model'].append(model)
                scalar_results['Screen'].append(screen)
                scalar_results['Eval Type'].append(et)
                scalar_results['Pearson'].append(corr)
                scalar_results['Concordance'].append(ci)
                
                
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

        # now do ground truth
        if screen!='Pooled':
            train_vals = pd.read_csv(f"../procdata/Broad/{drug}/data.csv",usecols = ['Cell Line','AAC'])
            test_vals = pd.read_csv(f"../procdata/{screen}/{drug}/data.csv",usecols = ['Cell Line','AAC'])
            # print(train_vals)
            gt = pd.merge(train_vals,test_vals,on='Cell Line')
            
            
            _km = min(gt.shape[0],kmax)
            

            krange = [x for x in kr if x<=_km]
            
            # krange = np.arange(1,_km+1,1)
            corr,_ = pearsonr(gt['AAC_x'].values,gt['AAC_y'].values)
            ci = evm.robust_concordance(gt['AAC_x'].values,gt['AAC_y'].values)
            scalar_results['Drug'].append(drug)
            scalar_results['Model'].append('Ground Truth')
            scalar_results['Screen'].append(screen)
            scalar_results['Eval Type'].append("Seen")
            scalar_results['Pearson'].append(corr)
            scalar_results['Concordance'].append(ci)


            for K in krange:
                        
                po = evm.perm_overlap(gt['AAC_y'].values,gt['AAC_x'].values,K)
                ndcg = evm.ndcg_at_k(gt['AAC_y'].values,gt['AAC_x'].values,K)
                ranking_results['Model'].append('Ground Truth')
                ranking_results['Drug'].append(drug)
                ranking_results['Screen'].append(screen)
                # baseline_results['Percentile'].append(p)
                ranking_results['K'].append(K)
                # baseline_results['Delta'].append(deltas[p])
                
                ranking_results['Eval Type'].append("Seen")
                ranking_results['NDCG'].append(ndcg)
                ranking_results['Perm Overlap'].append(po)
        


scalar_results = pd.DataFrame(scalar_results)
scalar_results.to_csv("../results/baseline_scalars.csv")
ranking_results = pd.DataFrame(ranking_results)
ranking_results.to_csv("../results/baseline_ranking.csv")
