#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:51:57 2021

@author: andrewchang
"""

import mne
import os


preload = True


proj_dir = os.getcwd()
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_4/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub4_teng/'
save_dir = proj_dir + '/save_fif/sub4_teng/'
subject = 'teng'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'FCT23_MPIEATENG_20180207_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'FCT23_MPIEATENG_20180207_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'FCT23_MPIEATENG_20180207_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'FCT23_MPIEATENG_20180207_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [2,13,34,39]