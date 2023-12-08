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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_17/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub17_Prisma_ALS23/'
save_dir = proj_dir + '/save_fif/sub17_ALS23/'
subject = 'ALS23'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'ALS23_MPIEATENG_20180508_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'ALS23_MPIEATENG_20180508_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'ALS23_MPIEATENG_20180508_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'ALS23_MPIEATENG_20180508_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [3,9,37,38,39]