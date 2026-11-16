#!/bin/bash

#SBATCH --time 3-00:00:00
#SBATCH --mem=30720M
#SBATCH -J param_sweep
#SBATCH -o ./slurmouts/param_sweep/%x-%j.out
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --array=0-20

drugs=(5-Fluorouracil AZD7762 AZD8055 Bortezomib Crizotinib Dabrafenib Dasatinib Docetaxel Erlotinib Gefitinib Gemcitabine Ibrutinib JQ1 Lapatinib MK-2206 Nilotinib PLX4720 Paclitaxel Pictilisib Vincristine Vorinostat AZD776)

source /cluster/home/t136576uhn/.bashrc
srun pixi run "python3 neural_param_sweep.py -config ./configs/neural/param_sweep.yaml -drug ${drugs[$SLURM_ARRAY_TASK_ID]}"