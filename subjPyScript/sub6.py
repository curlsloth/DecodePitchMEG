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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_6/'
dataMRI_dir = []
save_dir = proj_dir + '/save_fif/sub6_XWN16/'
subject = 'XWN16'
useTempMRI = 1
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation


raw = mne.io.read_raw_ctf(dataMEG_dir+'XWN16_MPIEATENG_20180409_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'XWN16_MPIEATENG_20180409_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'XWN16_MPIEATENG_20180409_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'XWN16_MPIEATENG_20180409_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [0,5,6,36,39]