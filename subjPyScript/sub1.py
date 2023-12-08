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
dataMEG_dir = proj_dir + '/Raw_MEG_data/subject_1/'
dataMRI_dir= proj_dir + '/HPC/MRI_image/sub1_Trio_DBK15/'
save_dir = proj_dir + '/save_fif/sub1_DBK15/'
subject = 'DBK15'
useTempMRI = 0

raw = mne.io.read_raw_ctf(dataMEG_dir+'DBK15_MPIEATENG_20180205_01_AUX.ds', preload=preload)
temp = mne.io.read_raw_ctf(dataMEG_dir+'DBK15_MPIEATENG_20180205_02_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'DBK15_MPIEATENG_20180205_03_AUX.ds', preload=preload)
raw.append(temp)
del temp
temp = mne.io.read_raw_ctf(dataMEG_dir+'DBK15_MPIEATENG_20180205_04_AUX.ds', preload=preload)
raw.append(temp)
del temp


ica_exclude = [22,28,31]