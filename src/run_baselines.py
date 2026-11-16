# ##
# 
# Code to run the baseline models against which 
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

	num_folds, drug_list, seed, write_dir, cv_score=\
		utils.unpack_parameters(config['EXPERIMENT_SETTINGS'])

	training_screen, testing_screens,  =\
		utils.unpack_parameters(config['DATA_SOURCES']) 
	

	model_to_hyperparams=\
		utils.unpack_parameters(config['MODELS_AND_HYPERPARAMETERS'],
			return_dict=True)

	

	# add the number of folds to each model for passing to 
	# sklearn pipeline
	for model_name in model_to_hyperparams.keys():
		model_to_hyperparams[model_name]['cv_obj'] = num_folds

	

	
	
	rng = np.random.default_rng(seed=seed)
	np.random.seed(seed=seed)
	rs = np.random.RandomState(seed=seed)
	
	# read in the list of drugs for testing
	drugs = utils.read_text_file(drug_list)

	

	# initialize dictionaries for storing results
	univariates = []
	
	# Try all drugs w/ a displayed progress bar. 
	
	for drug in (drug_bar:=tqdm.tqdm(drugs)):

		train_dir = f"{write_dir}/{drug}/baselines/train"
		os.makedirs(train_dir, exist_ok = True)

		test_dir = f"{write_dir}/{drug}/baselines/test"
		os.makedirs(test_dir, exist_ok = True)

		drug_bar.set_description(f"Working on {drug}")
		
		train_df = pd.read_csv(f"../procdata/{training_screen}/{drug}/data.csv")
		
		train_ccls = train_df['Cell Line']
		X_train = train_df[train_df.columns[1:-1]].values
		y_train = train_df[train_df.columns[-1]].values

	
		test_data = {}
		
		for test_screen in testing_screens:

			
			test_df = pd.read_csv(f"../procdata/{test_screen}/{drug}/data.csv")
			test_ccls = test_df['Cell Line']
			
		
			X_test = test_df[test_df.columns[1:-1]].values
			y_test = test_df[test_df.columns[-1]].values
			test_data[test_screen] = (test_ccls, X_test,y_test)

		

		for model_name in (pbar := tqdm.tqdm(model_to_hyperparams.keys(),leave=False)):
				
			pbar.set_description(f"Fitting {model_name}")
			
			
			
			# If it's a model implemented in scikit-learn we can use
			# that package's cross validation tooling. 
			
			if model_name in utils.SKLEARN_MODELS:

				# using the model name and hyperparameters fetch a
				# GridSearchCV object
				
				model = utils.build_model(model_name,**model_to_hyperparams[model_name],
							 random_state=rs)
				
				# fit the model via k-fold cross validation
				model.fit(X_train,y_train)

				train_preds = model.predict(X_train)
				
				if model_name =="Univariable":	
					gene = list(train_df.columns[model['select'].get_support(indices=True)+1])[0]
					univariates.append(f"{drug}\t{gene}\n")
				

				train_results = pd.DataFrame({"Cell Line":train_ccls, "Predicted":train_preds,"AAC":y_train})
				train_results.to_csv(f"{train_dir}/{model_name}.csv")

				# iterate over partitioned testing data and store the predictions
				# and ground truths
				
				for screen in test_data:
					ccls, X_test, y_test = test_data[screen]
					test_subdir = f"{test_dir}/{screen}"
					os.makedirs(test_subdir, exist_ok = True)
					preds = model.predict(X_test)
					test_results = pd.DataFrame({"Cell Line":ccls, "Predicted":preds, "AAC":y_test})
					test_results.to_csv(f"{test_subdir}/{model_name}.csv")

	with open(f"{write_dir}/univariable_markers.txt","w") as ostream:
		ostream.writelines(univariates)


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(
		prog = "Baseline models for DeepCINet.",
		description = "")

	parser.add_argument("-config", help = "The configuration file for the experiment.")

	args = parser.parse_args()
	with open(args.config) as config_file:
		config = yaml.safe_load(config_file)

	main(config)