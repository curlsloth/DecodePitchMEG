#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:52:05 2021

@author: andrewchang
"""

eventReCode_dict = {'flute_Ab6': 11, 'flute_C7': 12, 'flute_E7': 13, 'flute_Ab7': 14, 
                    'flute_C8': 15, 'flute_E8': 16, 'flute_Ab8': 17, 'flute_C9': 18, 
                    'keyboard_Ab6': 21, 'keyboard_C7': 22, 'keyboard_E7': 23, 'keyboard_Ab7': 24, 
                    'keyboard_C8': 25, 'keyboard_E8': 26, 'keyboard_Ab8': 27, 'keyboard_C9': 28, 
                    'string_Ab6': 31, 'string_C7': 32, 'string_E7': 33, 'string_Ab7': 34, 
                    'string_C8': 35, 'string_E8': 36, 'string_Ab8': 37, 'string_C9': 38}

# %% recodeEvents


def recodeEvents(events): 
    import numpy as np

    a = events[:,2].copy()
    for n in range(1,len(a)):
        if a[n] > 5:
            if a[n-1] == 1 and a[n]==7:
                a[n] = 11
            elif a[n-1] == 1 and a[n]==9:
                a[n] = 12
            elif a[n-1] == 1 and a[n]==11:
                a[n] = 13
            elif a[n-1] == 1 and a[n]==13:
                a[n] = 14
            elif a[n-1] == 1 and a[n]==15:
                a[n] = 15
            elif a[n-1] == 1 and a[n]==17:
                a[n] = 16
            elif a[n-1] == 1 and a[n]==19:
                a[n] = 17
            elif a[n-1] == 1 and a[n]==21:
                a[n] = 18
            elif a[n-1] == 2 and a[n]==7:
                a[n] = 21
            elif a[n-1] == 2 and a[n]==9:
                a[n] = 22
            elif a[n-1] == 2 and a[n]==11:
                a[n] = 23
            elif a[n-1] == 2 and a[n]==13:
                a[n] = 24
            elif a[n-1] == 2 and a[n]==15:
                a[n] = 25
            elif a[n-1] == 2 and a[n]==17:
                a[n] = 26
            elif a[n-1] == 2 and a[n]==19:
                a[n] = 27
            elif a[n-1] == 2 and a[n]==21:
                a[n] = 28
            elif a[n-1] == 3 and a[n]==7:
                a[n] = 31
            elif a[n-1] == 3 and a[n]==9:
                a[n] = 32
            elif a[n-1] == 3 and a[n]==11:
                a[n] = 33
            elif a[n-1] == 3 and a[n]==13:
                a[n] = 34
            elif a[n-1] == 3 and a[n]==15:
                a[n] = 35
            elif a[n-1] == 3 and a[n]==17:
                a[n] = 36
            elif a[n-1] == 3 and a[n]==19:
                a[n] = 37
            elif a[n-1] == 3 and a[n]==21:
                a[n] = 38

    events[:,2] = a
    events = np.delete(events,events[:,2]<10,0) # delete the triggers laballing the timbres
    events = np.delete(events,events[:,2]>100,0) # the trigger 128 labels the beginning of each sequence
    return events 






# %% preproICA

def preproICA(raw, save_dir, subject): 
    import mne
    import preprocFunc as pf    
    
    raw.load_data()
    raw_ica = raw.copy().filter(l_freq=1, h_freq=None, phase='zero-double') # ICA prefers higher highpass cutoff
    
    # downsample
    raw_ica.resample(sfreq=200)
    raw_ica.info
    
    # epoch
    events = mne.find_events(raw_ica)
    
    # recode events
    eventRecode = pf.recodeEvents(events)
    
    
    epochs_ica = mne.Epochs(raw_ica, eventRecode, event_id=eventReCode_dict, tmin=-0.5, tmax=1, 
                        baseline=(None, 0), preload=True, proj=False, picks='meg')


    epochs_ica.average().plot_joint() # just to check it visually
    
    ica = mne.preprocessing.ICA(n_components=40, random_state=23)
    ica.fit(epochs_ica)
    ica.save(save_dir+subject+'_ICA.fif')
        
    return epochs_ica, ica




# %% preproSource_freqbands


def preproSource_freqBands(raw, ica_exclude, save_dir, subject, dataMRI_dir, freq_bands, output_dir, runBEM=0, vis=0, useTempMRI=0, runDecode='basic_time'):

    import os.path as op
    # import numpy as np
    
    import mne
    from mne.datasets import fetch_fsaverage
    import pickle

    import preprocFunc as pf 
    
    n_jobs = -1
    
    ica = mne.preprocessing.read_ica(save_dir+subject+'_ICA.fif')
    raw.load_data()
    ica.exclude = ica_exclude # this is imported from the subject .py file
    
    ica.apply(raw)  
    
    
    
    for fband in freq_bands:
        
        
        reconst_raw = raw.copy()
          
    
        # filter data
        reconst_raw = reconst_raw.filter(l_freq=freq_bands[fband][0], h_freq=freq_bands[fband][1], phase='zero-double')
        reconst_raw.info
        
        
        # # downsample
        # reconst_raw.resample(sfreq=200)
        # reconst_raw.info
        # according to MNE, to avoid downsampling mess up with the trigger timing, it is recommended to (1) filter the data, and (2) decim the epoched data.
        # https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html#best-practices

        
        
        # epoch
        events = mne.find_events(reconst_raw)
        
        # recode events
        eventRecode = pf.recodeEvents(events)
        
        epochs_reconst = mne.Epochs(reconst_raw, eventRecode, event_id=eventReCode_dict, tmin=-0.2, tmax=0.5, 
                            baseline=(None, 0), preload=True, proj=False, picks='meg',
                            reject=dict(mag=4e-12),
                            decim = 6) # make the sampling rate as 200 Hz (original sf = 1200 Hz), which is > 3*40
        
        del reconst_raw
        
        epochs_reconst.average().plot_joint() # just to check it visually
        
        eventOut = epochs_reconst.events

        
        # %% compute cov matrix
    
        data_cov = mne.compute_covariance(epochs_reconst, tmin=0, tmax=None,
                                          method='empirical', n_jobs = n_jobs)
        noise_cov = mne.compute_covariance(epochs_reconst, tmin=None, tmax=0,
                                           method='empirical', n_jobs = n_jobs)
        data_cov.plot(epochs_reconst.info)
        
        # %% compute scr
    
        if useTempMRI == 0:
            src = mne.setup_source_space(subject, spacing='ico4', subjects_dir=dataMRI_dir, add_dist=False, n_jobs = n_jobs) # use ico4 to lower the spatial resolution
            # https://mne.tools/stable/overview/cookbook.html#setting-up-source-space
            # https://brainder.org/2016/05/31/downsampling-decimating-a-brain-surface/
        elif useTempMRI == 1: # if no subject-specific MRI, use template MRI
            fs_dir = fetch_fsaverage(verbose=True)
            src = op.join(fs_dir,'bem','fsaverage-ico-5-src.fif')
        
        # %% compute BEM
        
        if useTempMRI == 0:
            if runBEM == True:
                mne.bem.make_watershed_bem(subject, subjects_dir=dataMRI_dir, overwrite=True)
            
            bem_model = mne.make_bem_model(subject, subjects_dir=dataMRI_dir, conductivity=[0.3]) # specify only 1 number of conductivity only make 1 layer, which is sufficient for MEG
            bem = mne.make_bem_solution(bem_model)  # compute the bem solution
        elif useTempMRI == 1: # if no subject-specific MRI, use template MRI
            bem = op.join(fs_dir,'bem','fsaverage-5120-5120-5120-bem-sol.fif')

        
        # %% make fwd solution
    
        trans_file = save_dir + subject + '-trans.fif'
            
        fwd_all = mne.make_forward_solution(epochs_reconst.info, trans_file, src, bem,
                                            meg=True, n_jobs = n_jobs)
        
    
        
        
        # %% make inv solution
        inv_all = mne.minimum_norm.make_inverse_operator(epochs_reconst.info, fwd_all, noise_cov)
        
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        
        
        stc = mne.minimum_norm.apply_inverse_epochs(epochs_reconst, inv_all, lambda2, method='dSPM') # stc for the individual epochs

        # morph to the averaged brain
        if useTempMRI == 0:
            morph = mne.compute_source_morph(src, subject_from=subject, subject_to='fsaverage', subjects_dir=dataMRI_dir, spacing = 3) # spacing 3 = ico3
        else:
            morph = mne.compute_source_morph(src, subject_from='fsaverage', subject_to='fsaverage', spacing = 3)
            #del src
            
        for n in range(0, len(stc)):
            stc[n] = morph.apply(stc[n]) # replacing it
            
    
        # don's save the stc data. It is too big!!


    # %% call the decodeSourceMVPA function
        scores_FK, scores_FS, scores_KS, score_pitch, patterns_FK, patterns_FS, patterns_KS, patterns_pitch, score_tempGen_pitch = pf.decodeSourceMVPA(stc, eventOut)
        
        if (runDecode=='basic_time') or (runDecode=='both'):
            with open(output_dir+subject+'_'+fband+'_decode', 'wb') as fp:
                 pickle.dump([scores_FK, scores_FS, scores_KS, score_pitch, patterns_FK, patterns_FS, patterns_KS, patterns_pitch], fp)
                 fp.close()
                 
        if (runDecode=='time_gen') or (runDecode=='both'):
            with open(output_dir+subject+'_'+fband+'_decode_tempGen', 'wb') as fp:
                 pickle.dump(score_tempGen_pitch, fp)
                 fp.close()
    
    return 



# %% decodeSourceMVPA

def decodeSourceMVPA(stc, eventOut):
    #%%
    import mne
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import RidgeClassifierCV
    # from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
    from mne.decoding import (cross_val_multiscore, LinearModel, SlidingEstimator, GeneralizingEstimator,
                              get_coef)
    from sklearn.model_selection import StratifiedKFold
    
    #%% roi label
        
    mne.datasets.fetch_hcp_mmp_parcellation(verbose=True)
    
    combined = True # use the combined HCPMMP1 or not
    
    
    if combined:
        anat_label = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined')
        # extract the list of labels
        anat_label_list = []
        for n in range(0, len(anat_label)):
            #print(anat_label[n].name)
            anat_label_list.append(anat_label[n].name)
        
        ### select some regions
        roi = []
        roi = anat_label[anat_label_list.index('Early Auditory Cortex-lh')]
        roi += anat_label[anat_label_list.index('Early Auditory Cortex-rh')]
        roi += anat_label[anat_label_list.index('Auditory Association Cortex-lh')]
        roi += anat_label[anat_label_list.index('Auditory Association Cortex-rh')]
        roi += anat_label[anat_label_list.index('Inferior Frontal Cortex-lh')]
        roi += anat_label[anat_label_list.index('Inferior Frontal Cortex-rh')]
        roi += anat_label[anat_label_list.index('Insular and Frontal Opercular Cortex-lh')]
        roi += anat_label[anat_label_list.index('Insular and Frontal Opercular Cortex-rh')]
        roi += anat_label[anat_label_list.index('Orbital and Polar Frontal Cortex-lh')]
        roi += anat_label[anat_label_list.index('Orbital and Polar Frontal Cortex-rh')]
        roi += anat_label[anat_label_list.index('DorsoLateral Prefrontal Cortex-lh')]
        roi += anat_label[anat_label_list.index('DorsoLateral Prefrontal Cortex-rh')]
        
        
        
        # % morph to the averaged template brain
        
        roi_label_data = []
     
        for n in range(0, len(stc)):
            roi_label_data.append(stc[n].in_label(roi))
        
    else:
    
        anat_label = mne.read_labels_from_annot('fsaverage', 'HCPMMP1')
        
        
        # extract the list of labels
        anat_label_list = []
        for n in range(0, len(anat_label)):
            #print(anat_label[n].name)
            anat_label_list.append(anat_label[n].name)
        
        roi_A1 = []
        roi_beltAud = []
        roi_STG = []
        roi_STS = []
        # core auditory cortex (3 sources, if spacing = 3)
        roi_A1 = anat_label[anat_label_list.index('L_A1_ROI-lh')]
        roi_A1 += anat_label[anat_label_list.index('R_A1_ROI-rh')]
        
        
        # belt auditory cortex (19 sources, if spacing = 3)
        roi_beltAud = anat_label[anat_label_list.index('L_LBelt_ROI-lh')]
        roi_beltAud += anat_label[anat_label_list.index('L_MBelt_ROI-lh')]
        roi_beltAud += anat_label[anat_label_list.index('L_PBelt_ROI-lh')]
        roi_beltAud += anat_label[anat_label_list.index('L_RI_ROI-lh')]
        roi_beltAud += anat_label[anat_label_list.index('R_LBelt_ROI-rh')]
        roi_beltAud += anat_label[anat_label_list.index('R_MBelt_ROI-rh')]
        roi_beltAud += anat_label[anat_label_list.index('R_PBelt_ROI-rh')]
        roi_beltAud += anat_label[anat_label_list.index('R_RI_ROI-rh')]
    
        
        # auditory association areas (STG)
        roi_STG = anat_label[anat_label_list.index('L_A4_ROI-lh')]
        roi_STG += anat_label[anat_label_list.index('L_A5_ROI-lh')]
        roi_STG += anat_label[anat_label_list.index('L_STGa_ROI-lh')]
        roi_STG += anat_label[anat_label_list.index('L_TA2_ROI-lh')]
        roi_STG += anat_label[anat_label_list.index('R_A4_ROI-rh')]
        roi_STG += anat_label[anat_label_list.index('R_A5_ROI-rh')]
        roi_STG += anat_label[anat_label_list.index('R_STGa_ROI-rh')]
        roi_STG += anat_label[anat_label_list.index('R_TA2_ROI-rh')]
        
        # auditory association areas (STS)
        roi_STS = anat_label[anat_label_list.index('L_STSdp_ROI-lh')]
        roi_STS += anat_label[anat_label_list.index('L_STSda_ROI-lh')]
        roi_STS += anat_label[anat_label_list.index('L_STSva_ROI-lh')]
        roi_STS += anat_label[anat_label_list.index('L_STSvp_ROI-lh')]
        roi_STS += anat_label[anat_label_list.index('R_STSdp_ROI-rh')]
        roi_STS += anat_label[anat_label_list.index('R_STSda_ROI-rh')]
        roi_STS += anat_label[anat_label_list.index('R_STSva_ROI-rh')]
        roi_STS += anat_label[anat_label_list.index('R_STSvp_ROI-rh')]
        
        # all auditory regions: 66 sources, if spacing = 3
    
        # % morph to the averaged template brain
        
        roi_label_data = []
     
        for n in range(0, len(stc)):
            roi_label_data.append(stc[n].in_label(roi_A1+roi_beltAud+roi_STG+roi_STS))

    
    
    #%% build pipeline
    
    X = np.array([stc.data for stc in roi_label_data])
    
    
    
    del roi_label_data
    
    y = eventOut[:, 2].copy()
    
    n_jobs = -1
    
    
    n_splits=5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=23)
     
    # RidgeCV
    clf = make_pipeline(StandardScaler(),  # z-score normalization
                    SelectKBest(f_classif, k='all'),  # select features for speed
                    LinearModel(RidgeClassifierCV(alphas=np.logspace(-2,3,41), cv=cv)))
    
    
    
    # use this function to generate new estimator everytime
    def new_time_decode(clf, n_jobs):
        time_decod = SlidingEstimator(clf, n_jobs = n_jobs, scoring='roc_auc')
        return time_decod
    
    def new_timeGen_decode(clf, n_jobs):
        timeGen_decod = GeneralizingEstimator(clf, n_jobs = n_jobs, scoring='roc_auc')
        return timeGen_decod
    
    
    #%% pitch decoding
    y2 = y.copy()%10
    
    score_pitch = np.empty((n_splits,len(X[0,0,]),8,8))
    score_pitch[:] = np.NaN
    patterns_pitch = np.empty((len(X[0,]),len(X[0,0,]),8,8))
    patterns_pitch[:] = np.NaN
    score_tempGen_pitch = np.empty((n_splits,len(X[0,0,]),len(X[0,0,]),8,8))
    score_tempGen_pitch[:] = np.NaN
    
    
    nComb = 0
    from itertools import combinations
    for comb in combinations(range(1,9),2):
        time_decod_pitch = new_time_decode(clf, n_jobs)
        timeGen_decod_pitch = new_timeGen_decode(clf, n_jobs)
        nComb += 1
        p1 = comb[0]
        p2 = comb[1]
        X3 = X[np.logical_or(y2==p1, y2==p2),:,:].copy() # pitch the data which correspond to event p1 or p2
        y3 = y2[np.logical_or(y2==p1, y2==p2)].copy() # pitch the locations which correspond to event p1 or p2
        score_pitch[:,:,p1-1,p2-1] = cross_val_multiscore(time_decod_pitch, X3, y3, cv=cv, n_jobs=n_jobs)
        score_pitch[:,:,p2-1,p1-1] = score_pitch[:,:,p1-1,p2-1] # create an mirrored data
        
        score_tempGen_pitch[:,:,:,p1-1,p2-1] = cross_val_multiscore(timeGen_decod_pitch, X3, y3, cv=cv, n_jobs=n_jobs)
        score_tempGen_pitch[:,:,:,p2-1,p1-1] = score_tempGen_pitch[:,:,:,p1-1,p2-1]
        
        time_decod_pitch.fit(X3, y3)
        patterns_pitch[:,:,p1-1,p2-1] = get_coef(time_decod_pitch, "patterns_", inverse_transform=True)
        patterns_pitch[:,:,p2-1,p1-1] = patterns_pitch[:,:,p1-1,p2-1] # create an mirrored data
        

    #%% no longer in use, but keep them here for compatibility
    scores_FK = []
    scores_FS = []
    scores_KS = []
    patterns_FK = []
    patterns_FS = []
    patterns_KS = []
    
    #%% return variables
    return scores_FK, scores_FS, scores_KS, score_pitch, patterns_FK, patterns_FS, patterns_KS, patterns_pitch, score_tempGen_pitch




# %% seq_mlm_sepModels

def seq_mlm_sepModels(data, n_perm, corr_all):
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    
    import warnings
    warnings.filterwarnings("ignore")
    
    mdf_samePitch_list = []
    mdf_coch_list =[]

    for nt in range(data.shape[1]):
        # pitchDist = []
        pitchDist = pd.DataFrame()
        score_pitch_ind = data[:,nt,:,:]
            
        for n0 in range(np.shape(score_pitch_ind)[0]):
            for n1 in range(np.shape(score_pitch_ind)[1]):
                for ploc in range(n1+1,8):
                    # the unit of pitchDist has been modified to octave (Oct 17, 2023)
                    df = {'sub': n0, 'pitchDist': (ploc-n1)/3, 'samePitch': (ploc-n1==3 or ploc-n1==6), 'auc': score_pitch_ind[n0,n1,ploc], 'coch': corr_all[n1,ploc]}

                    pitchDist = pd.concat([pitchDist, pd.DataFrame([df])], ignore_index = True)
        
        pitchDist[['sub','samePitch']] = pitchDist[['sub','samePitch']].astype('category')
        
        if n_perm > 0:
            # randomly permute the 'auc' column within each sub. Needs 'sort_values()' to make sure 
            pitchDist[['auc','sub']] = pitchDist[['auc','sub']].groupby('sub').sample(frac=1, random_state=n_perm).reset_index(drop=True).sort_values(by=['sub'])


        md_samePitch = smf.mixedlm('auc ~ pitchDist + samePitch + pitchDist*samePitch', data=pitchDist, groups=pitchDist['sub'], re_formula='~pitchDist + samePitch + pitchDist*samePitch')
        md_coch = smf.mixedlm('auc ~ pitchDist + coch + pitchDist*coch', data=pitchDist, groups=pitchDist['sub'], re_formula='~pitchDist + coch + pitchDist*coch')
        
        try: 
            mdf_samePitch_list.append(md_samePitch.fit(disp=False))
            mdf_coch_list.append(md_coch.fit(disp=False))
        except: # very few cases it will get singular
            import random
            for nRow in range(len(pitchDist)):
                pitchDist['auc'].iloc[nRow] += (random.random()-0.5)*0.0002 # if getting singular, add 0.001 jitter to the data 
            md_samePitch = smf.mixedlm('auc ~ pitchDist + samePitch + pitchDist*samePitch', data=pitchDist, groups=pitchDist['sub'], re_formula='~pitchDist + samePitch + pitchDist*samePitch')
            md_coch = smf.mixedlm('auc ~ pitchDist + coch + pitchDist*coch', data=pitchDist, groups=pitchDist['sub'], re_formula='~pitchDist + coch + pitchDist*coch')
            mdf_samePitch_list.append(md_samePitch.fit(disp=False))
            mdf_coch_list.append(md_coch.fit(disp=False))
            
        
    
    df_samePitch_tvalues = pd.DataFrame()
    df_coch_tvalues = pd.DataFrame()
    df_samePitch_fe = pd.DataFrame()
    df_coch_fe = pd.DataFrame()
    
    for m in mdf_samePitch_list:
        df_samePitch_tvalues = pd.concat([df_samePitch_tvalues, pd.DataFrame([m.tvalues[:4]])], ignore_index=True) 
        if n_perm==0:
            df_samePitch_fe = pd.concat([df_samePitch_fe, pd.DataFrame([m.fe_params])], ignore_index=True) 
    for m in mdf_coch_list:
        df_coch_tvalues = pd.concat([df_coch_tvalues, pd.DataFrame([m.tvalues[:4]])], ignore_index=True) 
        if n_perm==0:
            df_coch_fe = pd.concat([df_coch_fe, pd.DataFrame([m.fe_params])], ignore_index=True) 
    
    return df_samePitch_tvalues, df_coch_tvalues, df_samePitch_fe, df_coch_fe



def seq_mlm_noiseCeiling(data, corr_all):
    import numpy as np
    import pandas as pd
    from scipy import stats
    import statsmodels.formula.api as smf
    
    import warnings
    warnings.filterwarnings("ignore")

    r_samePitch_lowbound_time = []
    r_samePitch_highbound_time = []
    r_coch_lowbound_time = []
    r_coch_highbound_time = []
    for nt in range(data.shape[1]):
        print('time: ', str(nt))
        # pitchDist = []
        pitchDist = pd.DataFrame()
        score_pitch_ind = data[:,nt,:,:]
            
        for n0 in range(np.shape(score_pitch_ind)[0]):
            for n1 in range(np.shape(score_pitch_ind)[1]):
                for ploc in range(n1+1,8):
                    # the unit of pitchDist has been modified to octave (Oct 17, 2023)
                    df = {'sub': n0, 'pitchDist': (ploc-n1)/3, 'samePitch': (ploc-n1==3 or ploc-n1==6), 'auc': score_pitch_ind[n0,n1,ploc], 'coch': corr_all[n1,ploc]}

                    pitchDist = pd.concat([pitchDist, pd.DataFrame([df])], ignore_index = True)
        
        pitchDist[['sub','samePitch']] = pitchDist[['sub','samePitch']].astype('category')
        
        r_samePitch_lowbound = []
        r_samePitch_highbound = []
        r_coch_lowbound = []
        r_coch_highbound = []
        for n_sub in range(data.shape[0]):
            # leave one out
            pitchDist_LOO = pitchDist[pitchDist['sub']!=n_sub]
            pitchDist_O = pitchDist[pitchDist['sub']==n_sub]
            
            for nc_bound in ['low','high']:
                
                if nc_bound == 'low':
                    df = pitchDist_LOO
                elif nc_bound == 'high':
                    df = pitchDist

                md_samePitch = smf.mixedlm('auc ~ pitchDist + samePitch + pitchDist*samePitch', data=df, groups=df['sub'], re_formula='~pitchDist + samePitch + pitchDist*samePitch')
                md_coch = smf.mixedlm('auc ~ pitchDist + coch + pitchDist*coch', data=df, groups=df['sub'], re_formula='~pitchDist + coch + pitchDist*coch')
                
                try: 
                    mdf_samePitch = md_samePitch.fit(disp=False)
                    mdf_coch = md_coch.fit(disp=False)
                except: # very few cases it will get singular
                    import random
                    for nRow in range(len(pitchDist)):
                        pitchDist['auc'].iloc[nRow] += (random.random()-0.5)*0.0002 # if getting singular, add 0.001 jitter to the data 
                    md_samePitch = smf.mixedlm('auc ~ pitchDist + samePitch + pitchDist*samePitch', data=df, groups=df['sub'], re_formula='~pitchDist + samePitch + pitchDist*samePitch')
                    md_coch = smf.mixedlm('auc ~ pitchDist + coch + pitchDist*coch', data=df, groups=df['sub'], re_formula='~pitchDist + coch + pitchDist*coch')
                    mdf_samePitch = md_samePitch.fit(disp=False)
                    mdf_coch = md_coch.fit(disp=False)
                
                fe_samePitch = mdf_samePitch.fe_params
                fe_coch = mdf_coch.fe_params
                
                pred_samePitch = fe_samePitch['Intercept'] + \
                    fe_samePitch['samePitch[T.True]']*(pitchDist_O['samePitch']==True) + \
                    fe_samePitch['pitchDist']*pitchDist_O['pitchDist'] + \
                    fe_samePitch['pitchDist:samePitch[T.True]']*(pitchDist_O['pitchDist'] * (pitchDist_O['samePitch']==True))
                pred_coch = fe_coch['Intercept'] + \
                    fe_coch['coch']*pitchDist_O['coch'] + \
                    fe_coch['pitchDist']*pitchDist_O['pitchDist'] + \
                    fe_coch['pitchDist:coch']*(pitchDist_O['pitchDist'] * pitchDist_O['coch'])
                
                r_samePitch, _ = stats.pearsonr(pred_samePitch, pitchDist_O['auc'])
                r_coch, _ = stats.pearsonr(pred_coch, pitchDist_O['auc'])
                
                if nc_bound == 'low':
                    r_samePitch_lowbound.append(r_samePitch)
                    r_coch_lowbound.append(r_coch)
                elif nc_bound == 'high':
                    r_samePitch_highbound.append(r_samePitch)
                    r_coch_highbound.append(r_coch)
                    
    
    r_samePitch_lowbound_time.append(np.mean(r_samePitch_lowbound))
    r_samePitch_highbound_time.append(np.mean(r_samePitch_highbound))
    r_coch_lowbound_time.append(np.mean(r_coch_lowbound))
    r_coch_highbound_time.append(np.mean(r_coch_highbound))
    
    return r_samePitch_lowbound_time, r_samePitch_highbound_time, r_coch_lowbound_time, r_coch_highbound_time

    

# %% return ERP
def get_ERP(raw, ica_exclude, save_dir, subject):
  
    import mne
    import preprocFunc as pf 
    
    
    ica = mne.preprocessing.read_ica(save_dir+subject+'_ICA.fif')
    raw.load_data()
    ica.exclude = ica_exclude # this is imported from the subject .py file
    
    ica.apply(raw)  
      
    reconst_raw = raw.copy()
      

    # filter data
    reconst_raw = reconst_raw.filter(l_freq=0.1, h_freq=40, phase='zero-double')
    reconst_raw.info
    
    
    # epoch
    events = mne.find_events(reconst_raw)
    
    # recode events
    eventRecode = pf.recodeEvents(events)
    
    epochs_reconst = mne.Epochs(reconst_raw, eventRecode, event_id=eventReCode_dict, tmin=-0.2, tmax=0.5, 
                        baseline=(None, 0), preload=True, proj=False, picks='meg',
                        reject=dict(mag=4e-12),
                        decim = 6) # make the sampling rate as 200 Hz (original sf = 1200 Hz), which is > 3*40
        
    return epochs_reconst