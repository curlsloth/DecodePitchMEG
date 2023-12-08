#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:41:36 2021

@author: andrewchang
"""

import os
import numpy as np
import mne
import preprocFunc as pf
import sys


######
## use the following few lines to check the input arguments
# print('Number of arguments:', len(sys.argv), 'arguments.')
# print('Argument List:', str(sys.argv))
# print('Type:', type(sys.argv))
## run the HPC with the code below
# sbatch --array=1-90 slurm_decode01b_RidgeCV_timeGen1031.s
######

input_arg = sys.argv[1]

file_num = int(np.ceil(int(input_arg)/5))
freq_num = int(int(input_arg)%5)

filename = 'subjPyScript/sub'+str(file_num)+'.py'
with open(filename, 'r') as file:
    code = file.read()

# Execute the code from the .py file
exec(code)


mne.datasets.fetch_fsaverage(subjects_dir="/scratch/ac8888/mne_data/MNE-fsaverage-data")
os.environ["SUBJECTS_DIR"] = "/scratch/ac8888/mne_data/MNE-fsaverage-data"



#%%


meg_channels = mne.pick_types(raw.info, meg=True)


# % get the source data

if freq_num==1:
    freq_bands = {'delta': (0.5,4)}
elif freq_num==2:
    freq_bands = {'theta': (4,8)}
elif freq_num==3:
    freq_bands = {'alpha': (8,13)}
elif freq_num==4:
    freq_bands = {'beta': (13,25)}
elif freq_num==0:
    freq_bands = {'gamma': (25,40)}
  

output_dir = 'sourceSTC20230711_ico3_freqBands_shuffled/decodeSource20230711_RidgeCV/auditory_frontal_alpha10^(-2)-10^3_41grid_correctPitchCoefPattern/'


pf.preproSource_freqBands(raw, ica_exclude, save_dir, subject, dataMRI_dir, freq_bands, output_dir, runBEM = 0, vis = 0, useTempMRI = useTempMRI, runDecode='time_gen')
# decoding procedure is embedded in it!

