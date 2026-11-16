import numpy as np
import sys
import pandas as pd
from collections import defaultdict

screens = ['Broad', 'GDSC1','GDSC2','gCSI']
with open("../procdata/drug_lists/post_qc.txt","r") as istream:
    drugs = istream.readlines()
    drugs = [x.rstrip() for x in drugs]

results = defaultdict(list)
for screen in screens:
    for drug in drugs:
        print(screen)
        print(drug)
        data = pd.read_csv(f"../procdata/{screen}/{drug}/data.csv")
        print(data)
        results['Screen'].append(screen)
        results['Drug'].append(drug)
        max_aac = np.round(np.amax(data['AAC'].values),3)
        min_aac = np.round(np.amin(data['AAC'].values),3)
        results['AAC Range'].append(f"[{min_aac},{max_aac}]")
        results['Num. CCLs'].append(len(pd.unique(data['Cell Line'])))


results = pd.DataFrame(results)
results.to_csv("../results/cell_counts_and_ranges.csv")

# print(drugs)

