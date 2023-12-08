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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_8/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub8_Prisma_NGP14/'
save_dir = proj_dir + '/save_fif/sub8_NGP14/'
subject = 'NGP14'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'NGS14_MPIEATENG_20180412_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'NGS14_MPIEATENG_20180412_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'NGS14_MPIEATENG_20180412_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'NGS14_MPIEATENG_20180412_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [0,10,20,34]