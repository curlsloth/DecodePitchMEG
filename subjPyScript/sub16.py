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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_16/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub16_Prisma_RAS12/'
save_dir = proj_dir + '/save_fif/sub16_RAS12/'
subject = 'RAS12'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'RAS12_MPIEATENG_20180503_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'RAS12_MPIEATENG_20180503_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'RAS12_MPIEATENG_20180503_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'RAS12_MPIEATENG_20180503_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [0,3,6,7,10,29,39]