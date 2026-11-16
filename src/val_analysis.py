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

neural_to_model = {'ECINET':"ECINET",'ONE_NARROW':"One Layer", "TWO_NARROW":"Two Layer"}


drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")

val_data = pd.read_csv("../results/neurals/param_sweep/val.csv",index_col=0)
_l = val_data[val_data['Screen']=='Broad']
_l = _l[(_l['Drug']=='Erlotinib')]
perf_data = pd.read_csv("../results/neurals/param_sweep/scalars.csv",index_col=0)
for model in pd.unique(_l['Model']):
    _ll = _l[_l['Model']==model]
    bd = _ll[_ll['Concordance']==np.max(_ll['Concordance'])]
    d = bd['Delta'].values[0]
    _p = perf_data[(perf_data['Screen']=='Pooled') & 
                   (perf_data['Model']==model) & 
                   (perf_data['Drug']=='Erlotinib')]
    print(_p[_p['Delta']==d])
sys.exit()




for split in [('gCSI','GDSC2'),('GDSC2','gCSI')]:
    val_set = split[0]
    perf = split[1]
    _v = val_data[val_data['Screen']==val_set]
    
    for model in pd.unique(_v['Model']):
        for drug in drugs:
            _m = _v[ (_v['Model']==model) & (_v['Drug']==drug) ]
            _p = perf_data[(perf_data['Screen']==perf)]
            _p = _p[(_p['Drug']==drug) & (_p['Model']==model)]
            
            ci = np.max(_m['Concordance'])
            print(ci)
            best_delta = _m[_m['Concordance']==ci]['Delta'].values[0]
            print(_p[_p['Delta']==best_delta])
            sys.exit()