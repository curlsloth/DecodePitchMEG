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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_11/'
dataMRI_dir = []
save_dir = proj_dir + '/save_fif/sub11_FCN23/'
subject = 'FCN23'
useTempMRI = 1
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation


raw = mne.io.read_raw_ctf(dataMEG_dir+'FCN23_MPIEATENG_20180423_01.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'FCN23_MPIEATENG_20180423_02.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'FCN23_MPIEATENG_20180423_03.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'FCN23_MPIEATENG_20180423_04.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [0,2,28,37]