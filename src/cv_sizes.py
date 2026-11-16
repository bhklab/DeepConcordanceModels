from collections import defaultdict
import torch
import pandas as pd
import numpy as np 
from DataClasses import PairwiseDataset
import utils
# import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from models import FFNN, ParsevalNet
from typing import List, Dict, Tuple
import argparse 
import time
import yaml
import os
import nn_utils
import sys
from models import TwinNet
from nn_utils import train_loop
import tqdm
# import datetime
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import StandardScaler


drug_to_delta = {
        '5-Fluorouracil':0.13,
        'AZD7762':0.0,
        'AZD8055':0.05,
        'Crizotinib':0.04,
        'Dasatinib':0.11,
        'Gefitinib':0.17,
        'Gemcitabine':0.15,
        'Lapatinib':0.2,
        'MK-2206':0.13,
         'Pictilisib':0.17,
         'Vorinostat':0.12
}

size_log = defaultdict(list)
for drug in drug_to_delta:
    train_df = pd.read_csv(f"../procdata/Broad/{drug}/data.csv")
    # print(train_df.shape)
    delta = drug_to_delta[drug]
    X = train_df[train_df.columns[1:-1]].values
    y = train_df[train_df.columns[-1]].values
    data_server = PairwiseDataset(X,y)
    data_server.update_data(delta=delta)
    size_log['Drug'].append(drug)
#             size_log['Delta'].append(delta)
    size_log['Num. Train Pairs'].append(len(data_server))

size_log = pd.DataFrame(size_log)
size_log.to_csv("../results/cv_sizes.csv")

