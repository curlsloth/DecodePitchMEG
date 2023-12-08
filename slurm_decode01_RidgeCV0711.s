#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=Andrew_RidgeCV0711
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --output=HPC/slurm/slurm_%A_%a.out

module purge
module load freesurfer/6.0.0 ; source $FREESURFER_HOME/SetUpFreeSurfer.sh

cd /scratch/ac8888/XiangbinMEG_music_decode

~/pyenv/run-mne.bash python decode01_HPC_prepro_source_decodeSource_freqBands.py $SLURM_ARRAY_TASK_ID