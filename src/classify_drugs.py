import pandas as pd
import numpy as np 
import utils
from collections import defaultdict

drugs = utils.read_text_file("../procdata/drug_lists/post_qc.txt")

known_targets = ['Bortezomib','Crizotinib','Dabrafenib','Dasatinib','Erlotinib',
                 'Gefitinib','Ibrutinib','Lapatinib','Pictilisib']
known_targets = []
res = defaultdict(list)
cut = 0.2

for screen in ['Broad','GDSC1','GDSC2','gCSI']:
    for drug in drugs:
    
        df = pd.read_csv(f"../procdata/{screen}/{drug}/data.csv",index_col=0)
        y = df['AAC']
        sig = [x  for x in y if x>=cut]
        mad = np.mean(np.abs(y-np.mean(y)))
    
        if len(sig)<=5:
            cl = 'No Effect'
        elif mad<=0.13:
            cl = 'Narrow Effect'
        else:
            cl = "Broad Effect"
        res['Drug'].append(drug)
        res['Screen'].append(screen)
        res['Class'].append(cl)


res = pd.DataFrame(res)
res.to_csv("../results/drug_classification.csv")
