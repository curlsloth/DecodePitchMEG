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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_10/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub10_Prisma_USE20/'
save_dir = proj_dir + '/save_fif/sub10_USE20/'
subject = 'USE20'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'USE20_MPIEATENG_20180416_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'USE20_MPIEATENG_20180416_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'USE20_MPIEATENG_20180416_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'USE20_MPIEATENG_20180416_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [11,34,36]