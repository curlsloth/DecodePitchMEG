#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:17:11 2021

@author: andrewchang
"""

import mne
import os

preload = True


proj_dir = os.getcwd()
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_13/'
dataMRI_dir = []
save_dir = proj_dir + '/save_fif/sub13_CHA21/'
subject = 'CHA21'
useTempMRI = 1
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation


raw = mne.io.read_raw_ctf(dataMEG_dir+'CHA21_MPIEATENG_20180423_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'CHA21_MPIEATENG_20180423_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'CHA21_MPIEATENG_20180423_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'CHA21_MPIEATENG_20180423_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [17,18,22,33,37]