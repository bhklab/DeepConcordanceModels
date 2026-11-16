import sys
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np  
import seaborn as sns
import utils 
from matplotlib_venn import venn3
import itertools as itl
from collections import defaultdict
import os
from scipy.special import comb





drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")

train_data = pd.read_csv("../procdata/Broad/responses.csv")
solids = utils.read_text_file("../procdata/Broad/solid_ccls.txt")

train_data = train_data[train_data['Cell Line'].isin(solids)]


gCSI = pd.read_csv("../procdata/gCSI/responses.csv")
solids = utils.read_text_file("../procdata/gCSI/solid_ccls.txt")
gCSI = gCSI[gCSI['Cell Line'].isin(solids)]

GDSC2 = pd.read_csv("../procdata/GDSC2/responses.csv")
solids = utils.read_text_file("../procdata/GDSC2/solid_ccls.txt")
GDSC2 = GDSC2[GDSC2['Cell Line'].isin(solids)]


res = defaultdict(list)
for drug in drugs: 
	_train = train_data[train_data['Drug']==drug]
	n_train = len(pd.unique(_train['Cell Line']))
	_t1 = gCSI[gCSI['Drug']==drug]
	_t1 = _t1[_t1['Cell Line'].isin(_train['Cell Line'])]
	_t2 = GDSC2[GDSC2['Drug']==drug]
	_t2 = _t2[_t2['Cell Line'].isin(_train['Cell Line'])]
	
	n_gcsi = len(pd.unique(_t1['Cell Line']))
	n_gdsc = len(pd.unique(_t2['Cell Line'])) 
	all_val = pd.concat((_t1, _t2), axis=0)
	n_val = len(pd.unique(all_val['Cell Line']))

	all_seen = pd.unique(all_val['Cell Line'])

	res['Drug'].append(drug)
	res['N. Train'].append(n_train)
	res['Train. Pairs'].append(int(comb(n_train,2)))
	res['N. Seen gCSI'].append(n_gcsi)
	res['N. Seen GDSC'].append(n_gdsc)
	res['N. Seen Val'].append(n_val)
	
	res['Val. Pairs'].append(int(comb(n_val,2)))

	_t1 = gCSI[gCSI['Drug']==drug]
	_t1 = _t1[~(_t1['Cell Line'].isin(_train['Cell Line']))]
	_t2 = GDSC2[GDSC2['Drug']==drug]
	_t2 = _t2[~(_t2['Cell Line'].isin(_train['Cell Line']))]


	all_test = pd.concat((gCSI[gCSI['Drug']==drug], GDSC2[GDSC2['Drug']==drug]), axis = 0)
	all_test = all_test[~(all_test['Cell Line'].isin(all_seen))]
	n_test = len(pd.unique(all_test['Cell Line']))
	n_gcsi = len(pd.unique(_t1['Cell Line']))
	n_gdsc = len(pd.unique(_t2['Cell Line'])) 
	res['N. Unseen gCSI'].append(n_gcsi)
	res['N. Uneen GDSC'].append(n_gdsc)
	res['N. Unseen Test'].append(n_test)
	res['K'].append(np.ceil(0.1*n_test))



res = pd.DataFrame(res)
print(res)
res.to_csv("../results/val_split_count.csv")
