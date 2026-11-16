from itertools import product
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
from sklearn.kernel_ridge import KernelRidge
import pandas as pd 
import tqdm
from typing import Dict, List
import utils
import argparse 
import yaml
import os
import sys
from datetime import datetime
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
from sklearn.metrics import r2_score
import yaml
import os
import sys
from evaluation_metrics import robust_concordance
from models import TwinNet
from nn_utils import train_loop, val_loop
import tqdm
# import datetime
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit


def main(config:Dict,drug:str):

	r"""
	As a first step we read in:
		- data sources
		- metadata
		- parameter ranges

		from the config file.
	""" 

	
	exp_category, ccl_type, drug_list, gene_list, seed, write_dir, use_probs, k_max =\
		utils.unpack_parameters(config['EXPERIMENT_SETTINGS'])


	ccl_col, drug_col, viability_col =\
		utils.unpack_parameters(config['INDEXING_INFO'])

	delta_min, delta_max, delta_step, delta_scale, extra_deltas, tau_values, tau_qtiles =\
		utils.unpack_parameters(config['PAIR_SELECTION'])

	num_epochs, batch_size, opt_algo, lrs, momentums, weight_decays, drop_outs, val_set =\
		utils.unpack_parameters(config['TRAIN_OPTS'])

	train_screen, train_expr_file, train_resp_file,train_ccl_file =\
		utils.unpack_parameters(config['TRAIN_DATA'])

	model_to_hyperparams=\
		utils.unpack_parameters(config['MODELS_AND_HYPERPARAMETERS'],
			return_dict=True)


	testing_data = utils.unpack_parameters(config['TEST_DATA'], return_dict = True)

	keep_genes = utils.read_text_file(gene_list)
	
	

	eval_data = {k:None for k in testing_data.keys()}
	
	for ds in testing_data.keys():
		test_ccls = utils.read_text_file(testing_data[ds]['test_ccl_file'])
		test_expr_file = testing_data[ds]['test_expr_file']
		test_resp_file = testing_data[ds]['test_resp_file']
		
		
		test_expr, test_resp = utils.fetch_and_subset_data(
			test_expr_file,
			test_resp_file,
			ccl_col,
			test_ccls,
			keep_genes)

		test_df = test_resp.merge(test_expr)

		eval_data[ds] = test_df


	
	# add the number of folds to each

	# gap_range = gap_scale*np.arange(gap_min, gap_max+1, gap_step)
	
	write_dir = f"{write_dir}/{val_set}"
	# print(write_dir)

	os.makedirs(write_dir, exist_ok = True)
	
	## we care only about ndcg at certain ranges of k values
	k_range = [k for k in range(1,k_max+1)]
	rng = np.random.default_rng(seed=seed)
	
	# read in the list of drugs for testing
	# drugs = utils.read_text_file(drug_list)

	if drug == "JQ1":
		drug = "JQ1 compound"
	drugs = [drug]
	

	train_ccls = utils.read_text_file(train_ccl_file)


	## Load the preprocessed CCL data
	
	train_expr,train_resp = utils.fetch_and_subset_data(
		train_expr_file,
		train_resp_file,
		ccl_col,
		train_ccls,
		keep_genes)


	deltas = np.concatenate((delta_scale*np.arange(delta_min, delta_max+1, delta_step),extra_deltas))
	# print(deltas)
	
	# add the number of folds to each

	# gap_range = gap_scale*np.arange(gap_min, gap_max+1, gap_step)
	gap_range = 10**-2*np.arange(0, 30, 1)
	
	os.makedirs(write_dir, exist_ok = True)

	## we care only about ndcg at certain ranges of k values
	k_range = [k for k in range(1,k_max+1)]
	rng = np.random.default_rng(seed=seed)
	
	# read in the list of drugs for testing
	# drugs = utils.read_text_file(drug_list)

	train_ccls = utils.read_text_file(train_ccl_file)


	## Load the preprocessed CCL data
	
	train_expr,train_resp = utils.fetch_and_subset_data(
		train_expr_file,
		train_resp_file,
		ccl_col,
		train_ccls,
		keep_genes)


	# initialize dictionaries for storing results

	ranking_results = defaultdict(list)
	scalar_results = defaultdict(list)
	size_log = defaultdict(list)

	# Try all drugs w/ a displayed progress bar. 
	# model_base = "/".join([write_dir,exp_category,"models"])
	# os.makedirs(model_base,exist_ok=True)


	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


	if use_probs:
		loss_fn = nn.BCELoss()
		loss_name = "BCE"
	else:
		loss_fn = nn.MarginRankingLoss()
		loss_name = "MRL"

	# initialize dictionaries for storing results

	ranking_results = defaultdict(list)
	scalar_results = defaultdict(list)
	size_log = defaultdict(list)

	# Try all drugs w/ a displayed progress bar.
	
	
	for drug in (drug_bar:=tqdm.tqdm(drugs)):
		hyper_params = defaultdict(list)
		
		drug_bar.set_description(f"Working on {drug}")

		res_dir = f"{write_dir}/{drug}"
		os.makedirs(res_dir,exist_ok=True)
		
		train_df = train_resp[train_resp[drug_col]==drug].merge(train_expr)
		
		_train_ccls = pd.unique(train_df[ccl_col])
	
		train_aacs = train_df[viability_col].values
	
		
		X_train, y_train = utils.build_data_set(
			train_df,
			keep_genes,
			ccl_col,
			viability_col)


		scaler = StandardScaler()

		X_train = scaler.fit_transform(X_train)

		tau_qs = [float( np.round(np.quantile(y_train,tq),2)) for tq in tau_qtiles]

		tau_names = ["".join([str(int(100*tq)),"%"]) for tq in tau_qtiles]
		val_df = pd.DataFrame()
		test_df = pd.DataFrame()
		

		for ds in eval_data.keys():
			

			# print(ds)
			test_data = eval_data[ds]
			test_data = test_data[test_data[drug_col]==drug]
			if val_set == "seen":

				# print(test_data.shape)
				# collect the cell lines which are in common 
				val_data = test_data[test_data[ccl_col].isin(_train_ccls)]
				# print(val_data.shape)
				val_df = pd.concat((val_df,val_data))

				test_data = test_data[~test_data[ccl_col].isin(_train_ccls)]

				test_df = pd.concat((test_df, test_data))

			else:
				if ds == val_set:
					val_df = pd.concat((val_df,test_data))
				else:
					test_df = pd.concat((test_df,test_data))



	

		X_val, y_val = utils.build_data_set(
			val_df,
			keep_genes,
			ccl_col,
			viability_col)


		X_val  = scaler.transform(X_val)

		

		drug_specific_taus = tau_qs

		train_server = PairwiseDataset(X_train,y_train,prob_loss = use_probs,symmetric_pairs=False)

		val_df = pd.DataFrame()
		test_df = pd.DataFrame()
		
		
		# X_val = preproc.transform(X_val)


		X_cv = np.vstack((X_train, X_val))
			
		y_cv = np.vstack((y_train.reshape(-1,1), y_val.reshape(-1,1))).reshape(-1,)

		val_server = PairwiseDataset(X_val,y_val, prob_loss = use_probs,delta=0, tau=0)
		
		cv_server = PairwiseDataset(X_cv,y_cv, prob_loss = use_probs,delta=0, tau=0)
		
		# test_server = PairwiseDataset(X_test,y_test, prob_loss = use_probs,delta=0, tau=0)

		# print(f"number train points: {X_train.shape}")
		# print(f"Number of val pairs: {len(train_server)}")
		# print(f"Number of val pairs: {len(val_server)}")
		# print(f"Number of cv pairs: {len(cv_server)}")
		# print(f"Number of test pairs: {len(test_server)}")
		

		val_loader = DataLoader(val_server, batch_size = batch_size, shuffle = False)
		

		for model_name in (pbar := tqdm.tqdm(model_to_hyperparams.keys(),leave=False)):
			pbar.set_description(f"Fitting Validating {model_name}")
			
			if model_name == "ECINET":
				_drop_probs = [0]
			else:
				_drop_probs = drop_outs
			
			param_grid = [
				(d.item(),t[0],t[1],lr,m,dp, wd) for (d,t,lr,m,dp,wd) in 
				product(deltas,zip(tau_qs,tau_names),lrs, momentums, _drop_probs,weight_decays)
				]
			
			
			
			for delta, tau, tq, lr, m, dp, wd in (param_bar := tqdm.tqdm(param_grid, 
				total = len(param_grid), leave=False)):
				param_bar.set_description(f"Working on: (delta ={delta}, tau = {tq},lr = {lr}, m = {m},d = {dp},wd = {wd})")
		
				hypers = model_to_hyperparams[model_name]
					
				hypers['prob_outs'] = use_probs
				hypers['drop_prob'] = dp
		
			
				train_server.update_data(delta=delta, tau=tau)

				train_loader = DataLoader(train_server, batch_size,shuffle=True)
				
				size_log['Drug'].append(drug)
				size_log['Delta'].append(delta)
				size_log['Tau'].append(tau)
				size_log['Num. Train Pairs'].append(len(train_server))



				hypers = model_to_hyperparams[model_name]	
				hypers['prob_outs'] = use_probs

				# model_dir = "/".join([model_base,drug,model_name,loss_name])
					
				# os.makedirs(model_dir,exist_ok=True)
					
				# fig_dir = "/".join([fig_base,model_name,loss_name])
				# os.makedirs(fig_dir,exist_ok = True)
					
				model = TwinNet(X_train.shape[1],**hypers).to(device)
				parseval_steps = False

				if opt_algo == 'sgd':
					optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum = m, weight_decay = wd)
				elif opt_algo == 'adam':
					optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay = wd)
				elif opt_algo == 'adamw':
					optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay = wd)
				
				
				scheduler = lr_scheduler.StepLR(optimizer, step_size=5)
				train_losses, val_losses = [], []
				train_ccs, val_ccs = [],[]
				
				for ep in (ep_bar:= tqdm.tqdm(range(num_epochs),leave=False)):
					epoch_loss = train_loop(model, device,train_loader,optimizer,use_probs,loss_fn,parseval_steps)
					curr_loss = epoch_loss/len(train_loader)

					
					val_loss = val_loop(model, device, val_loader, optimizer, use_probs, loss_fn)/len(val_loader)
					val_losses.append(val_loss)
					
					train_losses.append(curr_loss)

					if model_name !="ECINT":
						scheduler.step()

					with torch.no_grad():										
					
						preds = model.single_forward_pass(torch.tensor(X_val,dtype=torch.float32).to(device))	
						preds = preds.cpu().numpy().squeeze()
						val_ccs.append(robust_concordance(y_val, preds))
						
						val_conc = robust_concordance(y_val, preds)
						val_r2 = r2_score(y_val, preds)
						
						preds = model.single_forward_pass(torch.tensor(X_train,dtype=torch.float32).to(device))	
						preds = preds.cpu().numpy().squeeze()
						
						train_conc = robust_concordance(y_train, preds)
						train_ccs.append(robust_concordance(y_train, preds))
						train_r2 = r2_score(y_train, preds)

						


						hyper_params['Drug'].append(drug)
						hyper_params['Model'].append(model_name)
						hyper_params['Val. Data'].append(val_set)
						hyper_params['Epoch'].append(ep)
						hyper_params['Delta'].append(delta)
						hyper_params['Tau'].append(tau)
						hyper_params['Tau Name'].append(tq)
						hyper_params['Batch Size'].append(batch_size)
						hyper_params['Learning Rate'].append(lr)
						hyper_params['Weight Decay'].append(wd)
						hyper_params['Drop Out'].append(dp)
						hyper_params['Algorithm'].append(opt_algo)
						


						hyper_params['Train Loss'].append(curr_loss)
						hyper_params['Val Loss'].append(val_loss)
						hyper_params['Train Concordance'].append(train_conc)
						hyper_params['Val Concordance'].append(val_conc)
						hyper_params['Train r2'].append(train_r2)
						hyper_params['Val r2'].append(val_r2)

				# loss_fig_name = f"../figs/{drug}/Solid/{test_screen}_{loss_name}_{model_name}_Delta={delta}_Tau={tau}_training_log.png"
				# plt.plot(train_losses, label = "train")
				# plt.plot(val_losses, label = "val")
				# plt.xlabel("Epoch")
				# plt.ylabel("Loss")
				# plt.show()
				# plt.savefig(loss_fig_name)
				# plt.close()
				

		hyper_params = pd.DataFrame(hyper_params)
		hyper_params.to_csv(f"{res_dir}/param_search.csv")
				

					
			
				

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(
		prog = "Baseline models for DeepCINet.",
		description = "")

	parser.add_argument("-config", help = "The configuration file for the experiment.")
	parser.add_argument("-drug", help = "Drug to be analyzed.")
	args = parser.parse_args()
	with open(args.config) as config_file:
		config = yaml.safe_load(config_file)

	main(config,args.drug)
