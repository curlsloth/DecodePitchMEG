#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:39:26 2023

@author: andrewchang
"""

import os
import numpy as np
import pickle
import sys
n_perm_start = int(sys.argv[1])*10 # take input as n_perm random seed


#######
## run the HPC with the code below (array starting from 0 to run the original test)
# sbatch --array=0-500 slurm_decode02_permutation0717.s
#######

import preprocFunc as pf

subCode = ['DBK15',
           'SCN28', # no subject MRI
           'OKG26',
           'teng',
           'MEL23',
           'XWN16', # no subject MRI
           'ATE26',
           'NGP14',
           'KSS01',
           'USE20',
           # 'FCN23', # no subject; this is the 2nd scan of teng, skip it
           'EDK22',
           'CHA21', # no subject MRI
           'CBE22',
           'CTE02', 
           'RAS12',
           'ALS23',
           'EZE09']

subSel = np.array(range(0,len(subCode)))

timeAxis = np.round(np.arange(-0.2,0.505,0.005),3)



root_folder = 'sourceSTC20230711_ico3_freqBands_shuffled/decodeSource20230711_RidgeCV/'
file_folder_list = [
    root_folder+'auditory_frontal_alpha10^(-2)-10^3_41grid_correctPitchCoefPattern/'    
    ]

perm_folder = 'permutation_MLM_time_sepModels_reFull_spearmanCoch_17subjs'

for fband in ['alpha']:
    # run separate MLM models
    for file_folder in file_folder_list:
        scores_pitch_all = []
        
        for subject in subCode:
            
    
            # Random Forest Classifier: frequency bands
            with open(file_folder+subject+'_'+fband+'_decode', 'rb') as fp:
                _, _, _, score_pitch, _, _, _, _ = pickle.load(fp)  
                fp.close()
            
            scores_pitch_all.append(score_pitch)
            
            del score_pitch
        
        scores_pitch_all = np.stack([scores_pitch_all[i] for i in subSel], axis=0)
    
        data = np.mean(scores_pitch_all, axis = (1))
        corr_all = np.load('Stimuli_and_acoustics/cochleagram_db_spearman0723.npy')
        
        if n_perm_start > 0:
            for n_perm in range(n_perm_start, n_perm_start+10):
                df_samePitch_tvalues, df_coch_tvalues, _, _ = pf.seq_mlm_sepModels(data,n_perm,corr_all)
                df_samePitch_tvalues.set_index(timeAxis, inplace=True) # make index as time
                df_coch_tvalues.set_index(timeAxis, inplace=True) # make index as time
                if not os.path.exists(file_folder+perm_folder):
                    os.makedirs(file_folder+perm_folder)
                save_file_name_samePitch = file_folder+perm_folder+'/'+fband+'_samePitch_perm-tval-time'+'_'+str(n_perm)+'.csv'
                save_file_name_coch= file_folder+perm_folder+'/'+fband+'_coch_perm-tval-time'+'_'+str(n_perm)+'.csv'
                df_samePitch_tvalues.to_csv(save_file_name_samePitch)
                df_coch_tvalues.to_csv(save_file_name_coch)
                print(save_file_name_samePitch)
                print(save_file_name_coch)
        elif n_perm_start == 0:
            n_perm = n_perm_start
            df_samePitch_tvalues, df_coch_tvalues, df_samePitch_fe, df_coch_fe = pf.seq_mlm_sepModels(data,n_perm,corr_all)
            df_samePitch_tvalues.set_index(timeAxis, inplace=True) # make index as time
            df_coch_tvalues.set_index(timeAxis, inplace=True) # make index as time
            df_samePitch_fe.set_index(timeAxis, inplace=True) # make index as time
            df_coch_fe.set_index(timeAxis, inplace=True) # make index as time
            if not os.path.exists(file_folder+perm_folder):
                os.makedirs(file_folder+perm_folder)
            save_file_name_samePitch = file_folder+perm_folder+'/'+fband+'_samePitch_orig-tval-time.csv'
            save_file_name_coch = file_folder+perm_folder+'/'+fband+'_coch_orig-tval-time.csv'
            save_file_name_samePitch_fixedeffect = file_folder+perm_folder+'/'+fband+'_samePitch_orig-fixedeffect-time.csv'
            save_file_name_coch_fixedeffect = file_folder+perm_folder+'/'+fband+'_coch_orig-fixedeffect-time.csv'
            df_samePitch_tvalues.to_csv(save_file_name_samePitch)
            df_coch_tvalues.to_csv(save_file_name_coch)
            df_samePitch_fe.to_csv(save_file_name_samePitch_fixedeffect)
            df_coch_fe.to_csv(save_file_name_coch_fixedeffect)
            print(save_file_name_samePitch)
            print(save_file_name_coch)
            print(save_file_name_samePitch_fixedeffect)
            print(save_file_name_coch_fixedeffect)

