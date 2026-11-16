
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
neural_results = defaultdict(list)
for screen in screens:
    if screen=='Pooled':
        kmax = 20
    else:
        kmax = 10
    krange = np.concatenate(([1],np.arange(5,kmax+1,5)))
    for drug in (pbar:= tqdm.tqdm(drugs,leave=False)):  
        pbar.set_description(f"Working on {drug}")
        for model in neural_to_model:
            if screen == 'Pooled':
                s1  = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/test/GDSC2/{model}.csv",index_col=0)
                s2  = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/test/gCSI/{model}.csv",index_col=0)
                neural = pd.concat([s1,s2])
            else:
                neural = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/test/{screen}/{model}.csv",index_col=0)	
            train = pd.read_csv(f"../results/{drug}/neurals/replicate_gaps/train/{model}.csv",index_col=0)
            train_resp = train[['Cell Line','AAC']].drop_duplicates()
            cutoff = np.quantile(train_resp['AAC'].values,0.75)
            
            
            deltas = {}
            for k in pd.unique(neural['Percentile']):
                _t = neural[neural['Percentile']==k]
                deltas[str(k)] = _t['Delta'].values[0]
            
            
            
    

        
            
            neural = neural[neural['Epoch'].isin([ep])]
            train = train[train['Epoch'].isin([ep])]
            seen_cells = pd.unique(train['Cell Line'])


            for p in deltas:
                
                _temp = neural[neural['Percentile']==float(p)]
                seen = _temp[_temp['Cell Line'].isin(seen_cells)]
                unseen = _temp[~_temp['Cell Line'].isin(seen_cells)]
                

                
                
               
                for et, _df in zip(['Seen','Unseen'],[seen,unseen]):
                    preds = _df['Predicted'].values
                    y_true = _df['AAC'].values
                    y_binary = utils.make_ordinal(y_true,cutoff=cutoff)
                    num_rel = np.sum(y_binary)
                    
                   
                    for K in krange:
                       
                        po = evm.perm_overlap(y_true,preds,K)
                        ndcg = evm.ndcg_at_k(y_true, preds, K)
                        ndcg_bin = evm.ndcg_at_k(y_binary, preds, K)
                        if num_rel>0:
                            p_at_k = evm.precision_at_k(y_binary,preds,K)
                        else:
                            p_at_k = -1
                        
                        neural_results['Model'].append(neural_to_model[model])
                        neural_results['Drug'].append(drug)
                        neural_results['Screen'].append(screen)
                        neural_results['Percentile'].append(p)
                        neural_results['K'].append(K)
                        neural_results['Delta'].append(deltas[p])
                        
                        neural_results['Eval Type'].append(et)
                        neural_results['NDCG'].append(ndcg)
                        neural_results['Binary NDCG'].append(ndcg_bin)
                        neural_results['P@K'].append(p_at_k)
                        neural_results['Num. Rel'].append(num_rel)
                        neural_results['Perm Overlap'].append(po)                
neural_results = pd.DataFrame(neural_results)
res_dir = f"../results/neurals/"
os.makedirs(res_dir,exist_ok=True)
neural_results.to_csv(f"{res_dir}/ranking.csv")










screens = ['GDSC2','gCSI','GDSC1']

ep=30
baseline_results = defaultdict(list)
for screen in screens:
    if screen=='Pooled':
        kmax = 20
    else:
        kmax = 10
    krange = np.concatenate(([1],np.arange(5,kmax+1,5)))
    for drug in (pbar:= tqdm.tqdm(drugs,leave=False)):  
        pbar.set_description(f"Working on {drug}")
        for model in ['ElasticNet','RandomForest','Univariable']:
            if screen == 'Pooled':
                s1  = pd.read_csv(f"../results/{drug}/baselines/test/GDSC2/{model}.csv",index_col=0)
                s2  = pd.read_csv(f"../results/{drug}/baselines/test/gCSI/{model}.csv",index_col=0)
                res = pd.concat([s1,s2])
            else:
                res = pd.read_csv(f"../results/{drug}/baselines/test/{screen}/{model}.csv",index_col=0)	
            train = pd.read_csv(f"../results/{drug}/baselines/train/{model}.csv",index_col=0)
            

            

            # deltas = {}
            # for k in pd.unique(neural['Percentile']):
            #     _t = res[res['Percentile']==k]
            #     deltas[str(k)] = _t['Delta'].values[0]
            
            # neural = neural[neural['Epoch'].isin([ep])]
            # train = train[train['Epoch'].isin([ep])]
            seen_cells = pd.unique(train['Cell Line'])



          
            seen = res[res['Cell Line'].isin(seen_cells)]
            unseen = res[~res['Cell Line'].isin(seen_cells)]
                

                
                
               
            for et, _df in zip(['Seen','Unseen'],[seen,unseen]):
                
                for K in krange:
                        
                    po = evm.perm_overlap(_df['AAC'].values,_df['Predicted'].values,K)
                    ndcg = evm.ndcg_at_k(_df['AAC'].values,_df['Predicted'].values,K)
                    baseline_results['Model'].append(model)
                    baseline_results['Drug'].append(drug)
                    baseline_results['Screen'].append(screen)
                    # baseline_results['Percentile'].append(p)
                    baseline_results['K'].append(K)
                    # baseline_results['Delta'].append(deltas[p])
                    
                    baseline_results['Eval Type'].append(et)
                    baseline_results['NDCG'].append(ndcg)
                    baseline_results['Perm Overlap'].append(po)
baseline_results = pd.DataFrame(baseline_results)
res_dir = f"../results/"
os.makedirs(res_dir,exist_ok=True)
baseline_results.to_csv(f"{res_dir}/baselines_ranking.csv")