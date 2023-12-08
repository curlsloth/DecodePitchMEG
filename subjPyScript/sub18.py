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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_18/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub18_Prisma_EZE09/'
save_dir = proj_dir + '/save_fif/sub18_EZE09/'
subject = 'EZE09'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'EZE09_MPIEATENG_20180508_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'EZE09_MPIEATENG_20180508_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'EZE09_MPIEATENG_20180508_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'EZE09_MPIEATENG_20180508_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [32,38]