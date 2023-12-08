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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_14/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub14_Prisma_CBE22/'
save_dir = proj_dir + '/save_fif/sub14_CBE22/'
subject = 'CBE22'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'CBE22_MPIEATENG_20180503_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'CBE22_MPIEATENG_20180503_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'CBE22_MPIEATENG_20180503_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'CBE22_MPIEATENG_20180503_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [9,35,37,38]