# ##
# 
# Code to run the basic benchmarking models against which 
# 	we will measure out neural network approach. 
# 
# At a high level, the code works as follows: 
# 	Parameter settings are determined by a config file passed
#   as a command line argument. This file includes a experiment
#   settings, models to use, hyperparameter ranges, etc. Look 
#   at the config files for details on what the various parameters 
#   are. 
#
#
#
#
#			Author: James Bannon
#			Github: jbannon
####

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
from sklearn.preprocessing import StandardScaler



def main(config:Dict):

	r"""
	As a first step we read in:
		- data sources
		- metadata
		- parameter ranges

		from the config file.
	""" 

	num_folds, k_max, drug_list, seed, write_dir, ccl_type, gene_set,cv_score,thresholds =\
		utils.unpack_parameters(config['EXPERIMENT_SETTINGS'])

	training_screen, testing_screens,  =\
		utils.unpack_parameters(config['DATA_SOURCES']) 
	
	gap_min, gap_max, gap_step, gap_scale =\
		utils.unpack_parameters(config['EVAL_SETTINGS'])

	model_to_hyperparams=\
		utils.unpack_parameters(config['MODELS_AND_HYPERPARAMETERS'],
			return_dict=True)

	

	# add the number of folds to each
	
	for model_name in model_to_hyperparams.keys():
		model_to_hyperparams[model_name]['cv_obj'] = num_folds

	## gap range for rCI
	gap_range = gap_scale*np.arange(gap_min, gap_max+1, gap_step)
	
	write_dir = f"{write_dir}/{gene_set}/"
	os.makedirs(write_dir, exist_ok = True)



	## we care only about ndcg at certain ranges of k values
	k_range = [k for k in range(1,k_max+1)]
	rng = np.random.default_rng(seed=seed)
	
	
	# read in the list of drugs for testing
	drugs = utils.read_text_file(drug_list)

	

	# initialize dictionaries for storing results

	ranking_results = defaultdict(list)
	scalar_results = defaultdict(list)
	size_log = defaultdict(list)
	univariates = []
	# Try all drugs w/ a displayed progress bar. 
	for drug in (drug_bar:=tqdm.tqdm(drugs)):
		drug_bar.set_description(f"Working on {drug}")
		train_df = pd.read_csv(f"../procdata/{training_screen}/{ccl_type}/{gene_set}/{drug}/data.csv")
		
		X_train = train_df[train_df.columns[1:-1]].values
		y_train = train_df[train_df.columns[-1]].values

		thresh_to_cutoff = utils.make_threshold_map(y_train,thresholds)

		
		train_ccls = pd.unique(train_df[train_df.columns[0]])

		partitioned_test_data = {}
		for test_screen in testing_screens:

			partitioned_test_data[test_screen] = {}
			
			test_df = pd.read_csv(f"../procdata/{test_screen}/{ccl_type}/{gene_set}/{drug}/data.csv")
			test_ccls = pd.unique(test_df[test_df.columns[0]])
			


			# for analyzing the ground truth we keep the overlapping columns and merge
			# to make sure they're in the right order
			ccl_col, viability_col = train_df.columns[0], train_df.columns[-1]
			overlap_df = train_df[[ccl_col,viability_col]].merge(test_df[[ccl_col,viability_col]],on =ccl_col)
			

			# keep the values for later so that it follows the model syntax
			preds_gt = overlap_df[f"{viability_col}_x"].values
			y_test_gt = overlap_df[f"{viability_col}_y"].values
			
			seen_ccls = [x for x in test_ccls if x in train_ccls]
			unseen_ccls = [x for x in test_ccls if x not in train_ccls]

			for eval_type, ccl_subset in zip(
				['Seen','Unseen','All'], 
				[seen_ccls,unseen_ccls,None]
				):
				


				if ccl_subset is not None:
					data_subset = test_df[test_df[test_df.columns[0]].isin(ccl_subset)]

				X_test = data_subset[data_subset.columns[1:-1]].values
				y_test = data_subset[data_subset.columns[-1]].values

				if eval_type == "Seen":
					partitioned_test_data[test_screen][eval_type] = (X_test,y_test,preds_gt,y_test_gt)
				else:
					partitioned_test_data[test_screen][eval_type] = (X_test,y_test)


		
		for model_name in (pbar := tqdm.tqdm(model_to_hyperparams.keys(),leave=False)):
				
			pbar.set_description(f"Fitting {model_name}")
				
			# If it's a model implemented in scikit-learn we can use
			# that package's cross validation tooling. 
			
			model_results = {}
			
			if model_name in utils.SKLEARN_MODELS:

				# using the model name and hyperparameters fetch a
				# GridSearchCV object
				model = utils.build_model(model_name,**model_to_hyperparams[model_name])
				# scaler = StandardScaler()
				# scaler.fit(X_train)
				# fit the model via k-fold cross validation
				model.fit(X_train,y_train)
				if model_name =="Univariable":
					
					gene = list(train_df.columns[model['select'].get_support(indices=True)+1])[0]
					univariates.append(f"{drug}\t{gene}\n")
				# iterate over partitioned testing data and store the predictions
				# and ground truths
				for k in partitioned_test_data.keys():
					model_results[k] = {}
					splits = partitioned_test_data[k]
					for split_name in splits.keys():
						X_test, y_test = splits[split_name][0], splits[split_name][1]
						# X_test = scaler.transform(X_test)
						preds = model.predict(X_test)
						model_results[k][split_name] = (preds, y_test)

			elif model_name == "UniLasso":
				
				# in order to call the UniLasso fitting functions need to save the train/test
				# data as temp files
				# then we call the R script from this one
				np.savetxt("./train_data.csv",X_train,delimiter=",",fmt="%.18f",newline="\n")
				np.savetxt("./train_resp.csv",y_train,delimiter=",",fmt="%.18f",newline="\n")
				os.system(f"Rscript fit.uniLasso.R train_data.csv train_resp.csv model.rds {num_folds}")
				
				# evaluate the model across all testing screens
				# and the data split types
				
				for k in partitioned_test_data.keys():
					splits = partitioned_test_data[k]
					model_results[k] = {}
					for split_name in splits.keys():
						
						X_test, y_test = splits[split_name][0], splits[split_name][1]
						
						
						# same thing, need to save the data and call an R script
						# which loads the previously fit model
						np.savetxt("./test_data.csv",X_test,delimiter=",",fmt="%.18f",newline="\n")
						os.system("Rscript predict.unilasso.R test_data.csv model.rds preds.csv")

						preds = pd.read_csv("./preds.csv")
						preds = preds.iloc[:,-1].values
						model_results[k][split_name] = (preds, y_test)
	
				os.system("rm ./*.csv ./*.rds")
			
			elif model_name == "Random":
				
				
				for k in partitioned_test_data.keys():
					splits = partitioned_test_data[k]
					model_results[k] = {}
					for split_name in splits.keys():
						
						X_test, y_test = splits[split_name][0], splits[split_name][1]
						
						preds = rng.uniform(0,1,y_test.shape)
						
						model_results[k][split_name] = (preds, y_test)
	
			elif model_name == "GroundTruth":

				for k in partitioned_test_data.keys():
					splits = partitioned_test_data[k]
					model_results[k] = {}
					for split_name in ['Seen']:
						preds, y_test = splits[split_name][2], splits[split_name][3]
						model_results[k][split_name] = (preds, y_test)
					
					
					
					
			
			for ds in model_results.keys():
				
				for split in model_results[ds].keys():
					
					preds,y_test = model_results[ds][split][0],model_results[ds][split][1]
					
					utils.compute_scalar_metrics(
							res_dict = scalar_results,
							drug = drug,
							data_set = ds,
							model_name = model_name,
							eval_type = split,
							y_true = y_test,
							preds = preds,
							gap_range = gap_range,
							cv_score = cv_score)
					
					for thresh in thresh_to_cutoff.keys():
						cutoff = thresh_to_cutoff[thresh]
						
						utils.compute_ranking_metrics(
							res_dict = ranking_results,
							drug = drug,
							data_set = ds,
							model_name = model_name,
							eval_type = split,
							y_true = y_test,
							preds = preds,
							k_range = k_range,
							thresh=thresh,
							cutoff = cutoff,
							cv_score = cv_score)

					# for k in ranking_results.keys():
					# 	print(k)
					# 	print(len(ranking_results[k]))
					# sys.exit()



			
	cv_string = "CI" if cv_score is not None else "Loss"

	scalar_results = pd.DataFrame(scalar_results)
	ranking_results = pd.DataFrame(ranking_results)
	size_log = pd.DataFrame(size_log)

	scalar_results.to_csv(f"{write_dir}/scalar_metrics_{cv_string}.csv",index=False)
	ranking_results.to_csv(f"{write_dir}/ranking_metrics_{cv_string}.csv",index=False)
	with open(f"{write_dir}/univariable_markers.txt","w") as ostream:
		ostream.writelines(univariates)
	size_log.to_csv(f"{write_dir}/sizes.csv",index=False)

	now = datetime.now()
	now = now.strftime("%D %H:%M:%S")
	with open(f"{write_dir}/finish_time.txt","w") as ostream:
		ostream.writelines(now)







	



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(
		prog = "Baseline models for DeepCINet.",
		description = "")

	parser.add_argument("-config", help = "The configuration file for the experiment.")

	args = parser.parse_args()
	with open(args.config) as config_file:
		config = yaml.safe_load(config_file)

	main(config)