########
#
#	Script to run k-fold cross validation on the twin networks
# 	that make up DeepCINet. 
#
# 	
#   Specific experiment parameters are stored in a config file (.yaml)
#   that dictate which parameters are tested. 
#
#
#######


### imports  

# useful libraries 
from itertools import product
from collections import defaultdict
import tqdm
from typing import Dict, List, Tuple
import utils
import argparse 
import yaml
import os
import sys
from datetime import datetime



# scientific computing libraries
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit, KFold
from sklearn.metrics import r2_score

# our own concordance score
from evaluation_metrics import robust_concordance




# neural network imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler


# our neural functions
from models import TwinNet
import nn_utils
from DataClasses import PairwiseDataset

from scipy.stats import pearsonr



def main(config:Dict, drug:str):
	

	# correction to allow for use w/ slurm
	# there is almost definitely a better way
	if drug == "JQ1":
		drug = "JQ1 compound"
	
	

	# Read in:
	#	 The random generation seed for reproducibility
	# 	 The tissue type of cancer cell line (solid, liquid, any) 
	#	 The gene set under consideration. 
	#	 The largest k to consider for ranking metrics (e.g. largest k for precision @ k, etc.)
	#	 Threhold for binarizing AACs into sensitive/insensitive given as integer values 
	#		e.g., 1 => top 1% of AACs are called sensitive. 

	seed, write_dir, training_screen, testing_screens =\
		utils.unpack_parameters(config['EXPERIMENT_SETTINGS'])

	# Read in:
	#	The minimum and maximum value of delta as integers. 
	#   The scaling factor for delta
	#	Any additional (extra_deltas) (floats between 0 and 1)

	delta_min, delta_max, delta_step, delta_scale, extra_deltas =\
		utils.unpack_parameters(config['PAIR_SELECTION'])

	# Read In:
	#	The maximum number of epochs for training the neural network
	# 	The batch size of the neural network
	#	Which optimization algorithm to use
	# 	The learning rates to try for cross validation
	#	The momentum terms to try for cross validation
	# 	The weight decay values to try for cross validation
	#	The drop out values to try for cross validation
	# 	The number of splits of the data for training (i.e., the k for k-fold cross validatin)
	
	num_epochs, checkpts, batch_size, opt_algo, lrs, weight_decays, drop_outs, n_splits =\
		utils.unpack_parameters(config['TRAIN_OPTS'])


	model_to_hyperparams=\
		utils.unpack_parameters(config['MODELS_AND_HYPERPARAMETERS'],
			return_dict=True)


	rng = np.random.default_rng(seed=seed)
	rs = np.random.RandomState(seed=seed)
	torch.manual_seed(seed)

	


	loss_fn = nn.BCELoss()
	loss_name = "BCE"

	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	train_df = pd.read_csv(f"../procdata/{training_screen}/{drug}/data.csv")
		
	train_ccls = train_df['Cell Line'].values
	X = train_df[train_df.columns[1:-1]].values
	y = train_df[train_df.columns[-1]].values

	test_data = {}
			
	for test_screen in testing_screens:

		
		test_df = pd.read_csv(f"../procdata/{test_screen}/{drug}/data.csv")
		test_ccls = test_df['Cell Line']
		
	
		X_test = test_df[test_df.columns[1:-1]].values
		y_test = test_df[test_df.columns[-1]].values
		test_data[test_screen] = (test_ccls, X_test,y_test)
	


	train_dir = f"{write_dir}/{drug}/neurals/kfold/train/"
	os.makedirs(train_dir,exist_ok=True)

	test_dir = f"{write_dir}/{drug}/neurals/kfold/test/"
	os.makedirs(test_dir,exist_ok=True)
	
	deltas = np.round(np.concatenate((delta_scale*np.arange(delta_min, delta_max+1, delta_step),extra_deltas)),4)
	print(f"\n Working On: {drug}\n")

	for model_name in (pbar := tqdm.tqdm(model_to_hyperparams.keys(),leave=False)):
		pbar.set_description(f"Cross Validating {model_name}")

		
		
		param_grid = [
			(d.item(),lr,dp, wd) for (d,lr,dp,wd) in 
			product(deltas,lrs, drop_outs,weight_decays)
			]
		hyper_params = defaultdict(list)
		
		for delta, lr,  dp, wd in (param_bar := tqdm.tqdm(param_grid, total = len(param_grid), leave=False)):
			param_bar.set_description(f"Working on: (delta ={delta}, lr = {lr},d = {dp},wd = {wd})")
			splitter = KFold(n_splits = n_splits, shuffle = True, random_state = rs)

			hypers = model_to_hyperparams[model_name]
			hypers['drop_prob'] = dp
			for fold_num, (train_idx, val_idx) in (fold_bar:=tqdm.tqdm(enumerate(splitter.split(X,y)),leave=False,total=n_splits)):
				fold_bar.set_description(f"Working on fold {fold_num}")
				
				X_train, y_train = X[train_idx, :], y[train_idx]
				X_val, y_val = X[val_idx, :], y[val_idx]
				
				scaler = StandardScaler()

				X_train = scaler.fit_transform(X_train)
				X_val = scaler.transform(X_val)
				model = TwinNet(X_train.shape[1],**hypers).to(device)

				optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay = wd)
				train_server = PairwiseDataset(X_train,y_train, delta = delta)
				train_loader = DataLoader(train_server, batch_size, shuffle=True,drop_last=True)
				train_losses = []
		
				
				for ep in (ep_bar:= tqdm.tqdm(range(num_epochs),leave=False)):
					epoch_loss = nn_utils.train_loop(model, device,train_loader,optimizer,loss_fn)
					curr_loss = epoch_loss/len(train_loader)
					train_losses.append(curr_loss)

					if ep+1 in checkpts:
						with torch.no_grad():
							preds = model.single_forward_pass(torch.tensor(X_val,dtype=torch.float32).to(device))	
							preds = preds.cpu().numpy().squeeze()

							
						
							val_conc = robust_concordance(y_val, preds)

							
							if np.allclose(preds,preds[0]):
								val_corr = 0
							else:
								val_corr,_ = pearsonr(y_val,preds) 



							hyper_params['Drug'].append(drug)
							hyper_params['Model'].append(model_name)
							hyper_params['Epoch'].append(ep+1)
							hyper_params['Fold'].append(fold_num)
							hyper_params['Delta'].append(delta)
				
							hyper_params['Learning Rate'].append(lr)
							hyper_params['Weight Decay'].append(wd)
							hyper_params['Drop Out'].append(dp)
							hyper_params['Val Concordance'].append(val_conc)
							hyper_params['Val Pearson'].append(val_corr)
							
						
				
	

		
		
		hyper_params = pd.DataFrame(hyper_params)
		hyper_params.to_csv(f"{train_dir}/{model_name}_param_search.csv")
		summarized_params = hyper_params[['Epoch', 'Delta', 'Learning Rate', 'Weight Decay','Drop Out', 'Val Concordance', 'Val Pearson']]
		summarized_params = pd.DataFrame(summarized_params.groupby([ 'Epoch', 'Delta','Learning Rate', 'Weight Decay', 'Drop Out']).mean()).reset_index()
		

		best_ci = pd.DataFrame(summarized_params[summarized_params['Val Concordance']==np.max(summarized_params['Val Concordance'])])	
		best_corr = pd.DataFrame(summarized_params[summarized_params['Val Pearson']==np.max(summarized_params['Val Pearson'])])

		scaler = StandardScaler()

		X = scaler.fit_transform(X)
		

		named_param_pairs = zip(['ci','pearson'],[best_ci,best_corr])
		
		train_res = []
		test_res = defaultdict(list)
		print("cv ranges evaluated")
		for criteria,param_vals in named_param_pairs:
			# print(param_vals)
			_ep = param_vals['Epoch'].values[0]
			_d = param_vals['Delta'].values[0]
			_lr = param_vals['Learning Rate'].values[0]
			_wd = param_vals['Weight Decay'].values[0]
			_dp = param_vals['Drop Out'].values[0]


			hypers = model_to_hyperparams[model_name]
	
			hypers['drop_prob'] = _dp

			model = TwinNet(X.shape[1],**hypers).to(device)

			# print(_lr)

			optimizer = torch.optim.Adam(model.parameters(),lr=_lr,weight_decay = _wd)

			train_server = PairwiseDataset(X,y,delta = _d)
			train_loader = DataLoader(train_server, batch_size, shuffle=True,drop_last=True)
			train_losses = []
			
			for ep in (ep_bar:= tqdm.tqdm(range(_ep),leave=False)):
				ep_bar.set_description(f"Fitting Final {model_name} Model For {criteria}")
				
				epoch_loss = nn_utils.train_loop(model, device,train_loader,optimizer,loss_fn)
				curr_loss = epoch_loss/len(train_loader)
				train_losses.append(curr_loss)

		
			model.eval()
			
			with torch.no_grad():
				train_preds =  model.single_forward_pass(torch.tensor(X,dtype=torch.float32).to(device))
				train_preds = train_preds.cpu().numpy().squeeze()
				train_chkpt = pd.DataFrame({'Cell Line':train_ccls,
						'Predicted':train_preds, 
						'AAC':y,
						'Delta':_d,
						'Epoch':_ep,
						'Learning Rate':_lr,
						'Weight Decay':_wd,
						'Criteria':criteria})
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
						'Delta':_d,
						'Epoch':_ep,
						'Learning Rate':_lr,
						'Weight Decay':_wd,
						'Criteria':criteria})
					test_res[screen].append(test_chkpt)


		train_res = pd.concat(train_res)
		train_res.to_csv(f"{train_dir}/{model_name}.csv")

		for screen in test_res:
				
			test_subdir = f"{test_dir}{screen}/"
			os.makedirs(test_subdir,exist_ok=True)
			res_df = pd.concat(test_res[screen])
			res_df.to_csv(f"{test_subdir}/{model_name}.csv")
				

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
