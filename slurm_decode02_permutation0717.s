#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=Andrew_permutation0717
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --output=HPC/slurm/slurm_%A_%a.out

module purge

cd /scratch/ac8888/XiangbinMEG_music_decode

~/pyenv/run-mne.bash python decode02_HPC_pitchMLM_permutation.py $SLURM_ARRAY_TASK_ID