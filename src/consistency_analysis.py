import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_text_file
from evaluation_metrics import robust_concordance, random_concordance
from itertools import combinations
from collections import defaultdict

screenA = 'Broad'
screenB = 'GDSC1'

seed = 1234
drugs = read_text_file("../procdata/drug_lists/post_qc.txt")
delta = 0.03

screens = ['Broad','GDSC1','GDSC2','gCSI']

screen_pairs = combinations(screens,2)
results = defaultdict(list)
for s1,s2  in screen_pairs:
    for drug in drugs:     
        responseA = pd.read_csv(f"../procdata/{s1}/{drug}/data.csv",usecols= ["Cell Line", "AAC"])
        responseB = pd.read_csv(f"../procdata/{s2}/{drug}/data.csv",usecols= ["Cell Line", "AAC"])
        data = responseA.merge(responseB, on = "Cell Line")

        ci = robust_concordance(data['AAC_x'].values,data['AAC_y'].values)
       
        rci = robust_concordance(data['AAC_x'].values,data['AAC_y'].values,delta_o=delta,delta_p=delta)
        ranci = random_concordance(data['AAC_x'].values,data['AAC_y'].values)
        if rci==-1 or ci==-1:
            continue
        else:
            results['Drug'].append(drug)
            results['Measure'].append("CI")
            results['Performance'].append(ci)

            results['Drug'].append(drug)
            results['Measure'].append("rCI")
            results['Performance'].append(rci)

            results['Drug'].append(drug)
            results['Measure'].append("Random")
            results['Performance'].append(ranci)

results = pd.DataFrame(results)

avgs = results[['Measure','Performance']].groupby(by=['Measure']).mean()
avgs.to_csv("../results/average_rci.csv")

ax = sns.boxplot(results, x= 'Drug',y='Performance',hue='Measure',showfliers=False)
plt.xticks(rotation=60,ha='right')
plt.ylim(0.4, 1.05)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("../figs/consistency_boxplot.png")
plt.close()
sys.exit()


delta_range = (10**-2)*np.linspace(0,25,1000)

screen_pairs = [(x,y) for x,y in combinations(screens,2)]


for drug in drugs:
    results = defaultdict(list)
    for pair  in screen_pairs:
        s1 = pair[0]
        s2 = pair[1]
        # if s1 == 'GDSC1' and s2 == 'gCSI':
        #     continue
        # if s1 == "GDSC2" and s2 == "gCSI":
        #     continue
        
        responseA = pd.read_csv(f"../procdata/{s1}/{drug}/data.csv",usecols= ["Cell Line", "AAC"])
        responseB = pd.read_csv(f"../procdata/{s2}/{drug}/data.csv",usecols= ["Cell Line", "AAC"])
        data = responseA.merge(responseB, on = "Cell Line")

        # ci = robust_concordance(data['AAC_x'].values,data['AAC_y'].values)

        for delta in delta_range:
            rci = robust_concordance(data['AAC_x'].values,data['AAC_y'].values,delta_o=delta,delta_p=delta)
            if rci==-1 or ci==-1:
                continue
            else:
                results['Drug'].append(drug)
                results['Delta'].append(delta)
                results['Performance'].append(rci)
                results['Comparison'].append(f"{s1} vs. {s2}")

                # results['Drug'].append(drug)
                # results['Measure'].append("rCI")
                # results['Performance'].append(rci)
    
    results = pd.DataFrame(results)
    print(drug)
    print(results)
    sns.lineplot(results,x='Delta',y='Performance',hue= "Comparison",errorbar=None)
    plt.title(f"{drug} rCI vs. Delta")
    plt.savefig(f"../figs/{drug}_delta_curves.png")
    plt.close()