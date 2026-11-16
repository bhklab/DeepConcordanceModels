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
import yaml
import os
import nn_utils
import sys
from models import TwinNet
from nn_utils import train_loop
import tqdm
import time
# import datetime
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import StandardScaler


def main(
		config:Dict,
		drug:str):


	if drug == "JQ1":
		drug = "JQ1 compound"


	seed, write_dir,training_screen, testing_screens,replicate_type, drug_list,chkpts, quantiles, drop_types =\
		utils.unpack_parameters(config['EXPERIMENT_SETTINGS'])


	num_epochs, batch_size, opt_algo, lr, momentum, weight_decay =\
		utils.unpack_parameters(config['TRAIN_OPTS'])

	model_to_hyperparams=\
		utils.unpack_parameters(config['MODELS_AND_HYPERPARAMETERS'],
			return_dict=True)
	
	
	
	
	rng = np.random.default_rng(seed=seed)
	rs = np.random.RandomState(seed=seed)
	torch.manual_seed(seed)
	
	drugs = [drug]
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	



	replicates = pd.read_csv(f"../results/qc/CTRPv2_replicates.csv",index_col=0)
	
	info = pd.read_csv(f"../procdata/Broad/PSet_Data/ccl_data.csv")
	info = info[~info['tissueid'].isin(drop_types)]
	
	replicates = replicates[replicates['CCL'].isin(info['sampleid'])]
	
	replicates = replicates.dropna(subset=['Delta'])
	delta_range = [(0,0.0)]
	for qt in quantiles:
		
		q = np.quantile(np.abs(replicates['Delta']),qt/100)
		delta_range.append((str(qt),q))


	
	loss_fn = nn.BCELoss()
	loss_name = "BCE"

	

	for drug in drugs:
		train_df = pd.read_csv(f"../procdata/{training_screen}/{drug}/data.csv")
		
		train_ccls = train_df['Cell Line'].values
		X = train_df[train_df.columns[1:-1]].values
		y = train_df[train_df.columns[-1]].values
		
		
		
		scaler = StandardScaler()
		X = scaler.fit_transform(X)

		train_dir = f"{write_dir}/{drug}/neurals/replicate_gaps/train/"
		os.makedirs(train_dir,exist_ok=True)

		test_dir = f"{write_dir}/{drug}/neurals/replicate_gaps/test/"
		os.makedirs(test_dir,exist_ok=True)

		test_data = {}
			
		for test_screen in testing_screens:

			
			test_df = pd.read_csv(f"../procdata/{test_screen}/{drug}/data.csv")
			test_ccls = test_df['Cell Line']
			
		
			X_test = test_df[test_df.columns[1:-1]].values
			y_test = test_df[test_df.columns[-1]].values
			test_data[test_screen] = (test_ccls, X_test,y_test)
		


		print(f"Working on {drug}\n")
		size_log = defaultdict(list)
		for model_name in (pbar := tqdm.tqdm(model_to_hyperparams.keys(),leave=False)):
			pbar.set_description(f"Running Parameter Sweep for {model_name}")

			train_res = []
			test_res = defaultdict(list)

			data_server = PairwiseDataset(X,y)
			all_losses = []
			
			for delta_pair in (delta_bar:=tqdm.tqdm(delta_range,leave=False)):
				pct, delta = delta_pair
				delta_bar.set_description(f"Working on delta = {delta}")

				data_server.update_data(delta=delta)
				
				
				size_log['Drug'].append(drug)
				size_log['Delta Percentile'].append(pct)
				size_log['Delta'].append(delta)
				size_log['Num. Train Pairs'].append(len(data_server))
		
					
				hypers = model_to_hyperparams[model_name]
				train_loader = DataLoader(data_server, batch_size,shuffle=True,drop_last = True)
				model = TwinNet(X.shape[1],**hypers).to(device)
				
				optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay = weight_decay)
					
				
				losses = []
				
				for ep in (ep_bar:= tqdm.tqdm(range(num_epochs),leave=False)):
					epoch_loss = nn_utils.train_loop(model, device,train_loader,optimizer,loss_fn)
					losses.append(epoch_loss/len(train_loader))

					if ep+1 in chkpts:
						with torch.no_grad():
							train_preds =  model.single_forward_pass(torch.tensor(X,dtype=torch.float32).to(device))
							train_preds = train_preds.cpu().numpy().squeeze()
							train_chkpt = pd.DataFrame({'Cell Line':train_ccls,
									'Predicted':train_preds, 
									'AAC':y,
									'Percentile':pct,
									'Delta':delta,
									'Epoch':ep+1})
							train_res.append(train_chkpt)
							
							for screen in test_data:
			
								test_ccls, X_test, y_test = test_data[screen]
								X_test = scaler.transform(X_test)
								preds = model.single_forward_pass(torch.tensor(X_test,dtype=torch.float32).to(device))
								preds = preds.cpu().numpy().squeeze()
								test_chkpt = pd.DataFrame(
									{'Cell Line':test_ccls,
									'Predicted':preds, 
									'AAC':y_test,
									'Percentile':pct,
									'Delta':delta,
									'Epoch':ep+1})
								test_res[screen].append(test_chkpt)
							
				
				loss_df = pd.DataFrame({"Model":model_name,"Loss":losses,'Delta':delta, "Percentile":pct})					
				all_losses.append(loss_df)

			train_res = pd.concat(train_res)
			all_losses = pd.concat(all_losses)

			train_res.to_csv(f"{train_dir}/{model_name}.csv")
			all_losses.to_csv(f"{train_dir}/{model_name}_losses.csv")
			for screen in test_res:
				
				test_subdir = f"{test_dir}{screen}/"
				os.makedirs(test_subdir,exist_ok=True)
				res_df = pd.concat(test_res[screen])
				res_df.to_csv(f"{test_subdir}/{model_name}.csv")
					
			

		size_log = pd.DataFrame(size_log)
		size_log = size_log.drop_duplicates()
		size_log.to_csv(f"{write_dir}/{drug}/neurals/replicate_gaps/sizes.csv")



		
		

	



					

				
				
			
		
			
			




if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(
		prog = "Baseline models for DeepCINet.",
		description = "")
	parser.add_argument("-config", help = "The configuration file for the experiment.")
	parser.add_argument("-drug", help = "The drug to consider.")

	args = parser.parse_args()
	
	with open(args.config) as config_file:
		config = yaml.safe_load(config_file)

	main(config,args.drug)