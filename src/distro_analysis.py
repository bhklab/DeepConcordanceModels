import ot 
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import utils
import numpy as np 
from collections import defaultdict
import os
import sys


bins = np.linspace(0,1,10)
centroids = [ bins[i]+(bins[i+1]-bins[i])/2 for i in range(len(bins)-1)]
M = np.zeros( (len(centroids),len(centroids)))
for i in range(len(centroids)):
    for j in range(len(centroids)):
        M[i,j] = (centroids[i]-centroids[j])**2


drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")
screens = ['GDSC2','gCSI']

eval_type = 'Seen'

bl = pd.read_csv("../results/baseline_scalars.csv",index_col=0)
n = pd.read_csv("../results/neurals/kfold_scalars.csv",index_col=0)
cl = pd.read_csv("../results/drug_classification.csv")
n = n[n['Model'].isin(['TWO_LAYER'])]
n['Model']= 'DeepCINet-CV'
rg = pd.read_csv("../results/neurals/scalars.csv",index_col=0)
rg['Model'] = ["DeepCINet-"+str(x) for x in rg['Percentile']]

fr = []
for screen in screens:
    for drug in drugs:
        _n = n[(n['Drug']==drug) & (n['Screen']==screen) & (n['Criteria']=='pearson')]
        _g = rg[(rg['Drug']==drug) & (rg['Screen']==screen) &(rg['Eval Type']==eval_type) & (rg['Percentile'].isin([0,50,95]))]
        _g['Model'] = ["DeepCINet-"+str(x) for x in _g['Percentile']]
        _b = bl[(bl['Drug']==drug) &
                (bl['Screen']==screen) &
                (bl['Eval Type']==eval_type)]
        
      
        src = pd.read_csv(f"../procdata/Broad/{drug}/data.csv",usecols=['Cell Line','AAC'])
        tgt = pd.read_csv(f"../procdata/{screen}/{drug}/data.csv",usecols=['Cell Line','AAC'])

        src_dist,_ = np.histogram(src['AAC'].values,bins=bins)
        tgt_dist,_ = np.histogram(tgt['AAC'].values,bins=bins)
        
        src_dist = src_dist/np.sum(src_dist)
        
        tgt_dist = tgt_dist/np.sum(tgt_dist)
        d = ot.emd2(src_dist,tgt_dist,M)
       
        _fr = pd.concat([_n,_g,_b])
        _fr = _fr[['Drug', 'Model','Screen','Eval Type', 'Pearson','Concordance']]
        _fr['Wasserstein Distance'] = d
        _fr['Skew'] = skew(src_dist)
        _fr['Kurt'] = kurtosis(src_dist)
        fr.append(_fr)


fr = pd.concat(fr)
# fr = fr[fr['Model'].isin(['DeepCINet-95','DeepCINet-50','DeepCINet-0','DeepCINet-CV'])]
fr = fr[fr['Screen']=='gCSI']
print(fr)

for m in pd.unique(fr['Model']):
    _r = fr[fr['Model']==m]
    # print(_r)
    print("\n")
    print(m)
    # print(_r[['Concordance','Kurt']].corr())
    print(_r[['Pearson','Skew']].corr())
    # print(_r[['Concordance','Wasserstein Distance']].corr())
    print("\n")
#     sys.exit()    
# sns.scatterplot(fr,x='Wasserstein Distance',y='Pearson',hue= 'Model')
# plt.show()

# for drug in drugs:
    
#     Broad = pd.read_csv(f"../procdata/Broad/{drug}/data.csv",usecols=['Cell Line','AAC'])
#     gCSI = pd.read_csv(f"../procdata/gCSI/{drug}/data.csv",usecols=['Cell Line','AAC'])
#     GDSC2 = pd.read_csv(f"../procdata/GDSC2/{drug}/data.csv",usecols=['Cell Line','AAC'])

#     print(Broad)
#     broad_hist,_ = np.histogram(Broad['AAC'].values,bins=bins)
#     gcsi_hist, _ = np.histogram(gCSI['AAC'].values,bins=bins)
#     gdsc_hist, _ = np.histogram(GDSC2['AAC'].values,bins=bins)

#     print(broad_hist)
    
#     sys.exit()