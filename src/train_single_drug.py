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
from typing import List, Dict, Tuple
import argparse 
import time
import yaml
import os
import sys
from models import TwinNet, SimpleParsNet
from nn_utils import train_loop
import tqdm
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

def main(
	config:Dict,
	drug:str):


	ccl_type, drug_list, gene_list, seed, write_dir, use_probs, k_max =\
		utils.unpack_parameters(config['EXPERIMENT_SETTINGS'])


	ccl_col, drug_col, viability_col =\
		utils.unpack_parameters(config['INDEXING_INFO'])

	delta_min, delta_max, delta_step, delta_scale, extra_deltas, tau_values, tau_qtile =\
		utils.unpack_parameters(config['PAIR_SELECTION'])

	num_epochs, min_epochs,batch_size, opt_algo, lr, momentum, patience, stop_tol, zero_tol =\
		utils.unpack_parameters(config['TRAIN_OPTS'])

	train_screen, train_expr_file, train_resp_file,train_ccl_file =\
		utils.unpack_parameters(config['TRAIN_DATA'])

	model_to_hyperparams=\
		utils.unpack_parameters(config['MODELS_AND_HYPERPARAMETERS'],
			return_dict=True)
	
	testing_data = utils.unpack_parameters(config['TEST_DATA'], return_dict = True)
	
	train_ccls = utils.read_text_file(train_ccl_file)
	
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
	
	
	# read in the list of drugs for testing
	rng = np.random.default_rng(seed=seed)
	torch.manual_seed(seed)
	



	deltas = np.concatenate((delta_scale*np.arange(delta_min, delta_max+1, delta_step),extra_deltas))
	gap_range = 10**-2*np.arange(0, 30, 1)
	k_range = np.arange(1,k_max+1)
	

	

	
	if use_probs:
		loss_fn = nn.BCELoss()
		loss_name = "BCE"
	else:
		loss_fn = nn.MarginRankingLoss()
		loss_name = "MRL"
	
	model_base = "/".join([write_dir,"models"])

	os.makedirs(model_base,exist_ok=True)


	
	## Load the preprocessed CCL data
	
	train_expr,train_resp = utils.fetch_and_subset_data(
		train_expr_file,
		train_resp_file,
		ccl_col,
		train_ccls,
		keep_genes)
		

	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	

	
	

	fig_base = "/".join(["../figs",drug,'training_logs'])
	os.makedirs(fig_base,exist_ok= True)



	train_df = train_resp[train_resp[drug_col]==drug].merge(train_expr)

	_train_ccls = pd.unique(train_df[ccl_col])

	train_aacs = train_df[viability_col].values


	y = train_df[viability_col].values

	X = train_df[[g for g in keep_genes if g in train_df.columns]].values

	qcut = float( np.round(np.quantile(y,tau_qtile),2))
	
	drug_specific_taus = tau_values +[qcut]

	
	data_server = PairwiseDataset(X,y,prob_loss = use_probs)

	size_log = defaultdict(list)
	ranking_results = defaultdict(list)
	scalar_results = defaultdict(list)

	print(f"\nFitting neural nodels on: {drug}")

	res_path = f"../results/{drug}/neurals/"
	os.makedirs(res_path, exist_ok= True)
	
	for delta in (delta_bar:=tqdm.tqdm(deltas,leave=False)):
		delta_bar.set_description(f"Working on delta = {delta}")

		for tau in (tau_bar := tqdm.tqdm(drug_specific_taus,leave=False)):
			tau_bar.set_description(f"Working on tau = {tau}")
			data_server.update_data(delta=delta, tau=tau)

			train_loader = DataLoader(data_server, batch_size,shuffle=True)
			size_log['Drug'].append(drug)
			size_log['Delta'].append(delta)
			size_log['Tau'].append(tau)
			size_log['Num. Train Pairs'].append(len(data_server))
	
			for model_name in (pbar := tqdm.tqdm(model_to_hyperparams.keys(),leave=False)):
				
				pbar.set_description(f"Fitting {model_name}")
				hypers = model_to_hyperparams[model_name]
				
				hypers['prob_outs'] = use_probs

				model_dir = "/".join([model_base,drug,model_name,loss_name])
				
				os.makedirs(model_dir,exist_ok=True)
				
				fig_dir = "/".join([fig_base,model_name,loss_name])
				os.makedirs(fig_dir,exist_ok = True)
				if model_name[:4] == "PARS":
					model = SimpleParsNet(X.shape[1],**hypers).to(device)
					parseval_steps = True
				else:
					model = TwinNet(X.shape[1],**hypers).to(device)
					parseval_steps = False

				# print(model)                           
				if opt_algo == 'sgd':
					optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum = momentum)
				elif opt_algo == 'adam':
					optimizer = torch.optim.Adam(model.parameters(),lr = lr)
				elif opt_algo == 'adamw':
					optimizer = torch.optim.AdamW(model.parameters(),lr = lr)
				
				# scheduler = lr_scheduler.StepLR(optimizer, step_size=20)
				losses = []
				loss_diffs = []
				last_loss = 0
				
				

				


				early_stop = False
				for ep in (ep_bar:= tqdm.tqdm(range(num_epochs),leave=False)):
					epoch_loss = train_loop(model, device,train_loader,optimizer,use_probs,loss_fn,parseval_steps)
					curr_loss = epoch_loss/len(train_loader)
					losses.append(curr_loss)
					
					# if model_name!="ECINET":
					# 	scheduler.step()
					
					
					loss_gap = np.abs(curr_loss - last_loss)
					loss_diffs.append(loss_gap)
					
					last_loss = curr_loss
					

					
					if ep>=min_epochs and np.mean(loss_diffs[-patience:])<stop_tol:
						# print("setting early stop true")
						early_stop = True

					if ep>=min_epochs and curr_loss <= zero_tol:
						early_stop = True


				torch.save(model,f"{model_dir}/model_d={delta}_t={tau}.pth")

				loss_fig_name = f"{fig_dir}/Delta={delta}_Tau={tau}_training_log.png"
				plt.plot(losses)
				plt.xlabel("Epoch")
				plt.ylabel("Loss")
				plt.savefig(loss_fig_name)
				plt.close()



				model.eval()
				with torch.no_grad():
					for ds in eval_data.keys():
						test_df = eval_data[ds]
						test_df = test_df[test_df[drug_col]==drug]
						
						_test_ccls = pd.unique(test_df[ccl_col])
						
						seen_ccls = [x for x in _test_ccls if x in _train_ccls]
						unseen_ccls = [x for x in _test_ccls if x not in _train_ccls]

						for eval_type, subset in zip(
							['Seen','Unseen','All'],
							[seen_ccls,unseen_ccls,None]):
						
						
							X_test, y_test = utils.build_data_set(
								test_df,
								keep_genes,
								ccl_col,
								viability_col,
								subset)
						
							preds = model.single_forward_pass(torch.tensor(X_test,dtype=torch.float32).to(device))
							
							preds = preds.cpu().numpy().squeeze()
						
							utils.compute_scalar_metrics(
								scalar_results,
								drug,
								ds,
								model_name,
								eval_type,
								y_test,
								preds,
								gap_range,
								delta,
								tau)
							
							utils.compute_ranking_metrics(
								ranking_results,
								drug,
								ds,
								model_name,
								eval_type,
								y_test,
								preds,
								k_range,
								delta,
								tau)
						
	scalar_results = pd.DataFrame(scalar_results)
	ranking_results = pd.DataFrame(ranking_results)
	size_log = pd.DataFrame(size_log)


	scalar_results.to_csv(f"{res_path}/scalar_metrics.csv",index=False)
	ranking_results.to_csv(f"{res_path}/ranking_metrics.csv",index=False)
	size_log.to_csv(f"{res_path}/sizes.csv",index=False)

	

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(
		prog = "Neural Network Models for Drug Response",
		description = "")

	parser.add_argument("--config", help = "The configuration file for the experiment.")
	parser.add_argument("--drug", help = "the drug to be trained and saved on")
	args = parser.parse_args()
	with open(args.config) as config_file:
		config = yaml.safe_load(config_file)
	drug = args.drug

	main(config,drug)