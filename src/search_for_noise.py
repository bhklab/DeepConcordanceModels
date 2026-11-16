import sys
import pandas as pd 
import numpy as np
import utils
import matplotlib.pyplot  as plt
import seaborn as sns
from collections import defaultdict
import evaluation_metrics as evm

from scipy.stats import pearsonr

delta = 0.05915102013903200
drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")
df = pd.read_csv("../results/qc/CTRPv2_replicates.csv",index_col=0)
df = df[df['Drug'].isin(drugs)]



df = df[df['CCL'].isin(["A-549","CAL-51"])]



drug_to_list = {}
drug_to_delta = {}
res = defaultdict(list)
for drug in pd.unique(df['Drug']):
    for ccl in pd.unique(df['CCL']):
        _t = df[(df['Drug']==drug) & (df['CCL']==ccl)]
        reps = set(_t['X.Coord']).union(set(_t['Y.Coord']))
        res['AAC'].extend(sorted(list(reps)))
        res['Drug'].extend(len(reps)*[drug])
        res['CCL'].extend(len(reps)*[ccl])
        
        drug_to_delta[drug]=np.mean(_t['Delta'].values)

res = pd.DataFrame(res)
res = res[res['Drug'].isin(drugs)]
res = res.dropna()
sns.scatterplot(res,x='AAC',y='Drug',style='CCL',hue='CCL',s=100)
# plt.axvspan(0.1-delta/2, 0.1+delta/2,alpha = 0.2, color = 'k')
# plt.axvline(0.1,color = 'k', linestyle='--')
plt.tight_layout()
plt.savefig("../figs/fig1/noise_fig.png",dpi=500)
plt.close()
sns.set_theme(rc={'figure.figsize':(10,3)})



for drug in ['Erlotinib', 'Gemcitabine']:
    
    res1 = res[res['Drug'].isin([drug])]
   

    ax = sns.scatterplot(res1,x='AAC',y='Drug',style = 'CCL',hue='CCL',legend=False,s=100)
    ax.set_xlim(0,1)
    # plt.show()
    # sys.exit()
    for drug in pd.unique(res1['Drug']):
        _t = res[(res['Drug']==drug) & (res['CCL']=="A-549")]
       
        ls = "--"
        clr = 'purple'
        
        est = np.median(_t['AAC'])
        plt.axvline(est,color = clr, linestyle=ls)
        
        plt.axvspan(est-delta/2, est+delta/2,alpha = 0.2, color = 'k')
        # plt.xlim(0,0.2)

        _t = res[(res['Drug']==drug) & (res['CCL']=="CAL-51")]
        est = np.median(_t['AAC'])
        ls = "--"
        clr = 'orange'
        plt.axvline(est,color = clr, linestyle=ls)
        
        plt.ylabel(None)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        plt.savefig(f"../figs/fig1/subdelta_{drug}.png",dpi=500)
        plt.close()

# res = res[res['Drug'].isin(['Erlotinib','Gefitinib'])]
# drug_to_delta['Erlotinib']

# sns.scatterplot(res,x='AAC',y='Drug',hue='Drug',legend=False)
# for drug in pd.unique(res['Drug']):
#     _t = res[res['Drug']==drug]
#     delta = drug_to_delta[drug]
#     if drug == 'Erlotinib':
#         ls= '-'
#     else:
#         ls = '--'
#     plt.axvline(np.median(_t['AAC']+delta/2), color='k',linestyle=ls)
#     plt.axvline(np.median(_t['AAC']-delta/2),color = 'k', linestyle=ls)
# plt.tight_layout()
# plt.savefig("../figs/egfr_delta.png",dpi=500)

# plt.close()

# # keeps = sorted([v for v in drug_to_delta.values()])[:5]
# # points=[k for k in drug_to_delta if drug_to_delta[k] in (keeps)]
# # print(keeps)
# # print(points)
# # # for k in points:

# # print([drug_to_list[k] for k in points])
# # # print([np.mean(drug_to_list[k]) for k in points])