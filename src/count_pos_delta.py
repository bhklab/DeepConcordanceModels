import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from scipy.stats import wilcoxon
import pandas as pd
from scipy.stats import mannwhitneyu,ttest_ind
import utils 
from collections import defaultdict


drugs = utils.read_text_file("../procdata/drug_lists/feasible_list.txt")

res = defaultdict(list)
for drug in drugs:
    df = pd.read_csv(f"../results/{drug}/neurals/kfold/train/TWO_LAYER.csv")
    df =df[df['Criteria']=='pearson']
    res['Drug'].append(drug)
    res['Delta'].append(df['Delta'].values[0])


df = pd.DataFrame(res)
df.to_csv("../results/kfold_delta_count.csv")