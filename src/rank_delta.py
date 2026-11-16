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




res = pd.read_csv("../results/neurals/ranking.csv",index_col=0)
drugs = utils.read_text_file("../procdata/drug_lists/post_qc.txt")
# res = res[res['Screen']=='Pooled']

res = res[res['Percentile'].isin([0,95])]
res = res[res['Eval Type']=='Seen']

fig_dir = "../figs/fig2/"
os.makedirs(fig_dir,exist_ok=True)

deltas = list(pd.unique(res['Percentile']))
improvements = defaultdict(list)

for screen in ['gCSI', 'GDSC2','GDSC1']:
    for model in ['Two Layer']:
        _d = res[(res['Model']==model) & (res['Screen']==screen)]
        _d['Percentile'] = [str(x) for x in _d['Percentile']]
        
        # _p = _d[_d['Percentile']=='95']
        _p = _d[_d['K'].isin([10])]
    

        

        


        
        for k in [5,10]:
            _d2 = _d[_d['K']==k]
            print(_d2)
            for metric in ['Perm Overlap', 'NDCG','P@K']:

                sns.barplot(_d2,y='Drug', x=metric,hue='Percentile')
                plt.title(f"{metric} at {k}\nSeen CCLs in {screen}")
                plt.tight_layout()
                plt.savefig(f"{fig_dir}/{screen}_{metric}_at_{k}_gap.png",dpi=500)
                plt.close()

                for drug in drugs:
                    zero_score = _d2[(_d2['Percentile']=="0") &(_d2['Drug']==drug)][metric].values[0]
                    delta_score= _d2[(_d2['Percentile']=="95")&(_d2['Drug']==drug)][metric].values[0]
                    
                    improvements['Drug'].append(drug)
                    improvements['Model'].append(model)
                    improvements['Metric'].append(metric)
                    improvements['Percentile'].append("95")
                
                    improvements['Zero Score'].append(zero_score)
                    improvements['Delta Score'].append(delta_score)
                    improvements['Gap'].append(delta_score-zero_score)
                    improvements['K'].append(k)
                # improvements['Substantial'].append(delta_score>=1.03*zero_score)
            

improvements = pd.DataFrame(improvements)
improvements['K'] = [str(x) for x in improvements['K']]


for metric in ['NDCG','Perm Overlap']:
    im = improvements[(improvements['Model']=='Two Layer')& (improvements['Metric']==metric)]
    sns.barplot(im,x='Gap',y='Drug', hue = 'K')
    plt.title(f"Gap at each K for {metric}")
    plt.tight_layout()
    plt.show()


res = pd.read_csv("../results/neurals/scalars.csv")

improvements = defaultdict(list)
for model in ['Two Layer']:
    for drug in drugs:
        _d = res[(res['Drug']==drug) & (res['Model']==model)]

        for et in ['Seen','Unseen']:
            for metric in ['Pearson','Concordance']:
                for dd in deltas[1:]:
                    print(_d)
                    print(model)
                    print(drug)
                    print(et)

                    zero_score = _d[(_d['Percentile']==0) & (_d['Eval Type']==et)][metric].values[0]
                    delta_score= _d[(_d['Percentile']==dd)  & (_d['Eval Type']==et)][metric].values[0]
                    improvements['Drug'].append(drug)
                    improvements['Model'].append(model)
                    improvements['Metric'].append(metric)
                    improvements['Percentile'].append(dd)
                    improvements['Eval Type'].append(et)
                    improvements['Zero Score'].append(zero_score)
                    improvements['Delta Score'].append(delta_score)
                    improvements['Gap'].append(delta_score-zero_score)
                    improvements['Substantial'].append(delta_score>=1.03*zero_score)



im = pd.DataFrame(improvements)
# im = im[im['Drug'].isin(drugs)]
# im = im[im['Substantial']==True]
# im['Percentile'] = [str(x) for x in im['Percentile']]
tl = im[im['Model']=='Two Layer']
# tl = tl[tl['Percentile']==95]
z = tl[tl['Metric']=='Pearson']
z = z[z['Percentile'].isin([0,95])]
z = z[z['Eval Type']=='Unseen']
z['Percentile']=[str(x) for x in z['Percentile']]

tl = tl[tl['Eval Type']=='Seen']

tl = tl[tl['Metric']=='Pearson']
labs = np.round(tl['Delta Score'].values,2)
tl['Positive Delta Performance'] = pd.cut(
    tl['Delta Score'],
    bins=[-1,0.2,0.3,0.5,1],
    labels = ["<0.2","0.2-0.3","0.3-0.5",">0.5"])
palette = {'<0.2': "#919191", '0.2-0.3':  "#CF8B8B",'0.3-0.5':  "#E46262", '>0.5': "#FF0000"}
ax = sns.barplot(tl[tl['Metric']=='Pearson'], x='Gap', y='Drug',hue='Positive Delta Performance',palette=palette)
# ax.bar_label(ax.containers[-1], labels=labs,fontsize=10)
ax.set(xlim=(-0.6,0.4))
plt.tight_layout()
# plt.show()
plt.savefig(f"{fig_dir}/pearson_gap.png")
plt.close()
# ax.bar_label(ax.containers[-1], fmt='Mean:\n%.2f', label_type='center')
# plt.tight_layout()

# plt.close()
# sns.catplot(tl[tl['Metric']=='Concordance'], x='Gap', y='Drug', col = 'Eval Type',kind = 'bar')
# plt.tight_layout()
# plt.savefig("../figs/two_layer_ci.png")
# plt.close()

# # im = im[im['Percentile']==95]
# # im = im[im['Model'].isin(['ECINET','Two Layer'])]
# # sns.catplot(im[im['Metric']=='Pearson'], x='Gap', y='Drug', col = 'Eval Type',kind = 'bar',hue='Model')
# # plt.tight_layout()

# # plt.savefig("../figs/all_pearson.png")
# # plt.close()
# # sns.catplot(im[im['Metric']=='Concordance'], x='Gap', y='Drug', col = 'Eval Type',kind = 'bar',hue= 'Model')
# # plt.tight_layout()
# # plt.savefig("../figs/all_ci.png")
# # plt.close()

# # for screen in ['Pooled']:
# # 	_im = im[(im['Screen']==screen)]
# # 	sns.catplot(_im,kind = 'bar', x = "Gap", y="Drug",row = "Eval Type", col = "Metric",hue='Percentile')
# # 	plt.tight_layout()
# # 	plt.savefig("../figs/cat_growth.png",dpi=500)
# # 	# plt.show()
# # 	plt.close()
# # 	_temp = df[df['Screen']==screen]
# # 	_temp['Delta_bin'] = ["Positive" if float(x)>0 else "Zero" for x in _temp['Percentile']]
# # 	for metric in ['Pearson','Concordance']:
# # 		plt.title(f"{screen} - {metric}")
# # 		sns.violinplot(_temp, x= 'Eval Type',y=metric,hue= "Delta_bin")
# # 		plt.show()
# # 	sys.exit()
