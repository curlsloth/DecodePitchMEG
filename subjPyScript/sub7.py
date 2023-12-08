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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_7/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub7_Prisma_ATE26/'
save_dir = proj_dir + '/save_fif/sub7_ATE26/'
subject = 'ATE26'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'ATE26_MPIEATENG_20180412_01.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'ATE26_MPIEATENG_20180412_02.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'ATE26_MPIEATENG_20180412_03.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'ATE26_MPIEATENG_20180412_04.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [0,1,34,39]