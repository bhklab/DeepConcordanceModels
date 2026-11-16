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
import sys
from models import TwinNet
from nn_utils import train_loop
import tqdm
# import datetime
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

def main(config:Dict):


	num_folds, k_max, ccl_type, thresholds, drug_list, gene_list, seed, write_dir, use_probs =\
		utils.unpack_parameters(config['EXPERIMENT_SETTINGS'])


	ccl_col, drug_col, viability_col =\
		utils.unpack_parameters(config['INDEXING_INFO'])

	delta_min, delta_max, delta_step, delta_scale, tau_values, tau_qtile =\
		utils.unpack_parameters(config['PAIR_SELECTION'])

	num_epochs, batch_size, optimizer, lr, momentum =\
		utils.unpack_parameters(config['TRAIN_OPTS'])

	train_screen, train_expr_file, train_resp_file,train_ccl_file =\
		utils.unpack_parameters(config['TRAIN_DATA'])
	

	test_screen, test_expr_file, test_resp_file, test_ccl_file =\
		utils.unpack_parameters(config['TEST_DATA'])	

	model_to_hyperparams=\
		utils.unpack_parameters(config['MODELS_AND_HYPERPARAMETERS'],
			return_dict=True)
	

	
	# read in the list of drugs for testing
	rng = np.random.default_rng(seed=seed)
	torch.manual_seed(seed)
	

	k_range = [k for k in range(1,k_max+1)]
	deltas = delta_scale*np.arange(delta_min, delta_max+1, delta_step)
	
	

	# read in the list of drugs for testing
	drugs = utils.read_text_file(drug_list)

	train_ccls = utils.read_text_file(train_ccl_file)
	
	test_ccls = utils.read_text_file(test_ccl_file)

	keep_genes = utils.read_text_file(gene_list)


	if use_probs:
		loss_fn = nn.BCELoss()
		loss_name = "BCE"
	else:
		loss_fn = nn.MarginRankingLoss()
		loss_name = "MRL"
	
	res_path = "/".join([write_dir,"neurals",test_screen,loss_name])

	os.makedirs(res_path,exist_ok=True)
	## Load the preprocessed CCL data
	
	train_expr,train_resp = utils.fetch_and_subset_data(
		train_expr_file,
		train_resp_file,
		ccl_col,
		train_ccls,
		keep_genes)
	
	test_expr, test_resp = utils.fetch_and_subset_data(
		test_expr_file,
		test_resp_file,
		ccl_col,
		test_ccls,
		keep_genes)		
	

	
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	

	ranking_results = defaultdict(list)
	scalar_results = defaultdict(list)
	size_log = defaultdict(list)
	
	
		

	for drug in (drug_bar:=tqdm.tqdm(drugs)):
		drug_bar.set_description(f"Working on  {drug}")
		train_df = train_resp[train_resp[drug_col]==drug].merge(train_expr)

		_train_ccls = pd.unique(train_df[ccl_col])

		train_aacs = train_df[viability_col].values

		thresh_to_cutoff = utils.make_threshold_map(train_aacs,thresholds)


		y = train_df[viability_col].values

		X = train_df[[g for g in keep_genes if g in train_df.columns]].values

		qcut = float( np.round(np.quantile(y,tau_qtile),2))
		
		drug_specific_taus = tau_values +[qcut]
		# drug_specific_taus = [x for x in drug_specific_taus if x<= qcut]
		# drug_specific_taus = tau_values
		
		test_df = test_resp[test_resp[drug_col]==drug].merge(test_expr)


		_test_ccls = pd.unique(test_df[ccl_col])
		
		seen_ccls = [x for x in _test_ccls if x in _train_ccls]
		unseen_ccls = [x for x in _test_ccls if x not in _train_ccls]

	
		
		

		data_server = PairwiseDataset(X,y,prob_loss = use_probs)

	
	
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

					
					if model_name == "PARS_NET":
						model = SimpleParsNet(X.shape[1],**hypers).to(device)
						parseval_steps = True
					elif model_name == "HYBRID_NET":
						model = HybridNet(X.shape[1],**hypers).to(device)
						parseval_steps = True
					else:
						model = TwinNet(X.shape[1],**hypers).to(device)
						parseval_steps = False

					# print(model)
					
					optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum = momentum)
					
					optimizer = torch.optim.Adam(model.parameters(),lr = lr)
					scheduler = lr_scheduler.StepLR(optimizer, step_size=20)
					losses = []
					for ep in (ep_bar:= tqdm.tqdm(range(num_epochs),leave=False)):
						epoch_loss = train_loop(model, device,train_loader,optimizer,use_probs,loss_fn,parseval_steps)
						losses.append(epoch_loss/len(train_loader))
						scheduler.step()
					
					loss_fig_name = f"../figs/{drug}/Solid/{test_screen}_{loss_name}_{model_name}_Delta={delta}_Tau={tau}_training_log.png"
					plt.plot(losses)
					plt.xlabel("Epoch")
					plt.ylabel("Loss")
					plt.savefig(loss_fig_name)
					plt.close()
					
					model.eval()

					with torch.no_grad():
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
								model_name,
								eval_type,
								y_test,
								preds,
								[delta],
								delta,
								tau)
							
							utils.compute_ranking_metrics(
								ranking_results,
								drug,
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

	# now = datetime.now()
	# now = now.strftime("%D %H:%M:%S")
	# with open(f"{res_path}/finish_time.txt","w") as ostream:
	# 	ostream.writelines(now)



					

				
				
			
		
			
			
			
	print("\n************\n\nyay\n************\n\n")
	sys.exit()



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(
		prog = "Neural Network Models for Drug Response",
		description = "")

	parser.add_argument("-config", help = "The configuration file for the experiment.")

	args = parser.parse_args()
	with open(args.config) as config_file:
		config = yaml.safe_load(config_file)

	main(config)