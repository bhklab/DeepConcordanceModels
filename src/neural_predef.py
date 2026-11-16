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


def main(config:Dict,
	drug:str):

	if drug == "JQ1":
		drug = "JQ1 compound"


	seed, ccl_type, gene_set, write_dir, k_max, thresholds =\
		utils.unpack_parameters(config['EXPERIMENT_SETTINGS'])


	delta_min, delta_max, delta_step, delta_scale, extra_deltas =\
		utils.unpack_parameters(config['PAIR_SELECTION'])


	training_screen, testing_screens  =\
		utils.unpack_parameters(config['DATA'])


	num_epochs, batch_size, opt_algo, lr, momentum, weight_decay =\
		utils.unpack_parameters(config['TRAIN_OPTS'])

	model_to_hyperparams=\
		utils.unpack_parameters(config['MODELS_AND_HYPERPARAMETERS'],
			return_dict=True)
	
	k_range = [k for k in range(1,k_max+1)]
	rng = np.random.default_rng(seed=seed)
	rs = np.random.RandomState(seed=seed)
	torch.manual_seed(seed)
	

	deltas = np.round(delta_scale*np.arange(delta_min, delta_max+1, delta_step),4)
	
	
	loss_fn = nn.BCELoss()
	loss_name = "BCE"

	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
	train_df = pd.read_csv(f"../procdata/{training_screen}/{ccl_type}/{gene_set}/{drug}/data.csv")
		
	X = train_df[train_df.columns[1:-1]].values
	y = train_df[train_df.columns[-1]].values
	
	thresh_to_cutoff = utils.make_threshold_map(y,thresholds)
	
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	
	gap_range = 10**-2*np.arange(0, 30, 1)

	train_ccls = pd.unique(train_df[train_df.columns[0]])

	partitioned_test_data = utils.fetch_partitioned_data(train_df, train_ccls, testing_screens,gene_set,drug, ccl_type)
	


	print(f"Working on {drug}\n")
	
	for model_name in (pbar := tqdm.tqdm(model_to_hyperparams.keys(),leave=False)):
		pbar.set_description(f"Running Parameter Sweep for {model_name}")
		
		res_dir = f"{write_dir}/{drug}/{gene_set}/{training_screen}/{model_name}"
		# print(res_dir)
		os.makedirs(res_dir,exist_ok=True)

		ranking_results = defaultdict(list)
		scalar_results = defaultdict(list)
		size_log = defaultdict(list)


	
		data_server = PairwiseDataset(X,y)

	
	
		for delta in (delta_bar:=tqdm.tqdm(deltas,leave=False)):
			delta_bar.set_description(f"Working on delta = {delta}")

			
			data_server.update_data(delta=delta)
			train_loader = DataLoader(data_server, batch_size,shuffle=True)
			
			size_log['Drug'].append(drug)
			size_log['Delta'].append(delta)
			size_log['Num. Train Pairs'].append(len(data_server))
	
				
			hypers = model_to_hyperparams[model_name]
				
				
			model = TwinNet(X.shape[1],**hypers).to(device)
					

				
			optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum = momentum)
				
			chkpts = [5,10,15,20,25,30]
			losses = []
			
			for ep in (ep_bar:= tqdm.tqdm(range(num_epochs),leave=False)):
				epoch_loss = nn_utils.train_loop(model, device,train_loader,optimizer,loss_fn)
				losses.append(epoch_loss/len(train_loader))

				if ep+1 in chkpts:
					with torch.no_grad():
						for ds in partitioned_test_data.keys():
							splits = partitioned_test_data[ds]
							for eval_type in splits.keys():
								X_test, y_test = splits[eval_type][0], splits[eval_type][1]
								X_test = scaler.transform(X_test)
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
									f"{ep}_epochs")
								

								for thresh in thresh_to_cutoff.keys():

									cutoff = thresh_to_cutoff[thresh]
									
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
										thresh,
										cutoff,
										f"{ep+1} Epochs")

				
			fig_dir = f"../figs/{drug}/training_logs/{training_screen}/{ccl_type}/{gene_set}/{model_name}"
			os.makedirs(fig_dir,exist_ok=True)
			loss_fig_name = f"{fig_dir}/Delta={delta}.png"
			plt.plot(losses)
			plt.xlabel("Epoch")
			plt.ylabel("Loss")
			plt.savefig(loss_fig_name)
			plt.close()
			
			
							


		scalar_results = pd.DataFrame(scalar_results)
		ranking_results = pd.DataFrame(ranking_results)
		size_log = pd.DataFrame(size_log)



		scalar_results.to_csv(f"{res_dir}/scalar_metrics.csv",index=False)
		ranking_results.to_csv(f"{res_dir}/ranking_metrics.csv",index=False)
		size_log.to_csv(f"{res_dir}/sizes.csv",index=False)

	# now = datetime.now()
	# now = now.strftime("%D %H:%M:%S")
	# with open(f"{res_path}/finish_time.txt","w") as ostream:
	# 	ostream.writelines(now)



					

				
				
			
		
			
			




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