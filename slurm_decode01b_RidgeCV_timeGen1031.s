#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=5:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=Andrew_RidgeCV_timeGen1031
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ac8888@nyu.edu
#SBATCH --output=HPC/slurm/slurm_%A_%a.out

module purge
module load freesurfer/6.0.0 ; source $FREESURFER_HOME/SetUpFreeSurfer.sh

cd /scratch/ac8888/XiangbinMEG_music_decode

~/pyenv/run-mne.bash python decode01b_HPC_prepro_source_decodeSource_freqBands_temporalGeneralization.py $SLURM_ARRAY_TASK_ID

