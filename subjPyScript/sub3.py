#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:08:12 2021

@author: andrewchang
"""


import mne
import os


preload = True


proj_dir = os.getcwd()
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_3/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub3_Trio_OKG26/'
save_dir = proj_dir + '/save_fif/sub3_OKG26/'
subject = 'OKG26'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'OKG26_MPIEATENG_20180207_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'OKG26_MPIEATENG_20180207_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'OKG26_MPIEATENG_20180207_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'OKG26_MPIEATENG_20180207_04_AUX.ds', preload=preload)
raw.append(temp)
del temp

ica_exclude = [6,16,17,18,33,38]