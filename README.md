# Deep Concordance Models (DeepCINET)

This repository contains code supporting the paper "Modelling Phenotypic Assay Noise for Multivariable Biomarker Discovery using Deep Neural Networks."





# Environment Set Up

Pixi is required to run this project. If you haven't installed it yet, follow [these instructions](https://pixi.prefix.dev/latest/). 

# Initialization

1. Clone this repository to your local machine
2. Navigate to the project directory
2. Set up the environment using Pixi:

```
pixi install
```


# Running Experiments 

First navigate to the `/src` directory.

## Generating the data

To prepare the data for the machine learning work first run 

`Rscript fetch_and_process_psets.R`
followed by 
`pixi run "python3 make_data.py"`



*Note* Because of weirdness with pixi at present (interactions with R and osx-arm64 hardware) you need to install required R packages in a conda environment or on your local machine. For convenience the pre-processed data are already included in this repository and so the above steps are optional. 

Additional utility scripts that need to be run:

- `pixi run "python3 process_ttd.py"` 


## QC Figures and Exploratory Analysis

The work on robust concordance index (rCI),  drug classification, and distributional analysis can be reproduced using the following commands:


- `pixi run "python3 plot_replicate_gaps.py"` to get the data and figures needed for figure 2, table 1, and the supplementary material. 

- `pixi run "python3 exploratory analysis"` to look at the AAC distribution within each and across screens. 
- `pixi run "python3 compute_aac_range_and_cell_counts.py` to curate at the dose ranges and AAC ranges for each drug in each screen. 
- `pixi run "python3 search_for_noise.py"` 
- 


## Experiments

### Baselines

Running the baseline experiments simply requires running:

`pixi run "python3 run_baselines.py -config ./configs/baselines.yaml"`

The neural network experiments were run using high performance computing clusters GPU acceleration managed with SLURM job management. In order to generate the results you need to run

- `pixi run "python3 run_neural_kfold.py -config ./configs/kfold.yaml -drug <drug>"`
and
- `pixi run "python3 run_neural_predef.py -config ./configs/kfold.yaml -drug <drug>"`

for each drug. 

As above the results from these have been pushed to the repository for full reproducibility. 


## After running neural models we run

- `pixi run "python3 analyze_neurals_scalar.py"`

- `pixi run "python3 analyze_neurals_ranking.py"`

- `pixi run "python3 seen_evaluation.py"`

- `pixi run "python3 count_pos_delta.py"`

- `pixi run "python3 kfold_analysis.py"`

- `pixi run "python3 kfold_ranking_analysis.py"`

