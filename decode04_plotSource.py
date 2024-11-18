#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:26:29 2023

@author: andrewchang
"""


import mne
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


file_folder = 'sourceSTC20230711_ico3_freqBands_shuffled/decodeSource20230711_RidgeCV/auditory_frontal_alpha10^(-2)-10^3_41grid_correctPitchCoefPattern/'


with open(file_folder+'vertices.pickle', 'rb') as handle:
    vertices = pickle.load(handle)
    handle.close()


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
           # 'FCN23', # no subject MRI
           'EDK22',
           'CHA21', # no subject MRI
           'CBE22',
           'CTE02', 
           'RAS12',
           'ALS23',
           'EZE09']

subSel = np.array(range(0,len(subCode)))

timeAxis = np.round(np.arange(-0.2,0.505,0.005),3)



patterns_pitch_all = []




for subject in subCode:

    
    fband = 'delta'
    with open(file_folder+subject+'_'+fband+'_decode', 'rb') as fp:
        _, _, _, _, _, _, _, patterns_pitch = pickle.load(fp)  
        fp.close()
        patterns_pitch_all.append(patterns_pitch)
    
patterns_pitch_all_array= np.stack([patterns_pitch_all[i] for i in subSel], axis=0)
patterns_pitch_all_array = np.array(patterns_pitch_all)


patterns_pitch_select_list = []

for n1 in range(8):
    for n2 in range(n1,8):
        if n2-n1 > 2: # only select the pairs which are at least 1 oct apart
            print([n1, n2])
            patterns_pitch_select_list.append(patterns_pitch_all_array[:,:,:,n1,n2])

patterns_pitch_select_array_ind= np.nanmean(np.stack([patterns_pitch_select_list[i] for i in range(15)], axis=0),axis=(0))
patterns_pitch_select_array= np.nanmean(np.stack([patterns_pitch_select_list[i] for i in range(15)], axis=0),axis=(0,1))



stc_feat = mne.SourceEstimate(
    np.abs(patterns_pitch_select_array),
    vertices=vertices,
    tmin=-0.2,
    tstep=0.005,
    subject="fsaverage",
)


data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + "/subjects"


# ROI list
anat_label = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined')
# extract the list of labels
anat_label_list = []
for n in range(0, len(anat_label)):
    #print(anat_label[n].name)
    anat_label_list.append(anat_label[n].name)

### select some regions
roi_list = []
roi_list.append(anat_label[anat_label_list.index('Early Auditory Cortex-lh')])
roi_list.append(anat_label[anat_label_list.index('Early Auditory Cortex-rh')])
roi_list.append(anat_label[anat_label_list.index('Auditory Association Cortex-lh')])
roi_list.append(anat_label[anat_label_list.index('Auditory Association Cortex-rh')])
roi_list.append(anat_label[anat_label_list.index('Inferior Frontal Cortex-lh')])
roi_list.append(anat_label[anat_label_list.index('Inferior Frontal Cortex-rh')])
roi_list.append(anat_label[anat_label_list.index('Insular and Frontal Opercular Cortex-lh')])
roi_list.append(anat_label[anat_label_list.index('Insular and Frontal Opercular Cortex-rh')])
roi_list.append(anat_label[anat_label_list.index('Orbital and Polar Frontal Cortex-lh')])
roi_list.append(anat_label[anat_label_list.index('Orbital and Polar Frontal Cortex-rh')])
roi_list.append(anat_label[anat_label_list.index('DorsoLateral Prefrontal Cortex-lh')])
roi_list.append(anat_label[anat_label_list.index('DorsoLateral Prefrontal Cortex-rh')])

label_list = ['Early Auditory', 'Auditory Association','Orbital and Polar Frontal','Insular and Frontal Opercular','Inferior Frontal','DorsoLateral Prefrontal']

# %% plot brain atlas

Brain = mne.viz.get_brain_class()
brain = Brain(
    "fsaverage",
    "split",
    "inflated",
    subjects_dir=subjects_dir,
    cortex="low_contrast",
    background="white",
    size=(1200, 600),
)

colors = sns.color_palette("Set2")
colors = sns.color_palette("deep", 10)
alpha = 0.9
for label in roi_list:
    if "Early Auditory Cortex" in label.name:
        brain.add_label(label, borders=False, color=colors[3], alpha=alpha)
    elif "Auditory Association Cortex" in label.name:
        brain.add_label(label, borders=False, color=colors[1], alpha=alpha)
    elif "Inferior Frontal Cortex" in label.name:
        brain.add_label(label, borders=False, color=colors[4], alpha=alpha)
    elif "Insular and Frontal Opercular Cortex" in label.name:
        brain.add_label(label, borders=False, color=colors[5], alpha=alpha)
    elif "Orbital and Polar Frontal Cortex" in label.name:
        brain.add_label(label, borders=False, color=colors[2], alpha=alpha)
    elif "DorsoLateral Prefrontal Cortex" in label.name:
        brain.add_label(label, borders=False, color=colors[0], alpha=alpha)
    
brain.show_view(col = 1, view={'elevation':90, 'azimuth':35})
brain.show_view(col = 0, view={'elevation':90, 'azimuth':145})
# brain.save_image("ROI_atlas.png")

# %% plot brain

def vis_brain(stc_feat, t_min, t_max, clim, view, subjects_dir=subjects_dir, roi_list=roi_list):
    brain = stc_feat.copy().crop(t_min, t_max).mean().plot(
        hemi='split',
        transparent=True,
        time_unit="s",
        subjects_dir=subjects_dir,
        clim={'kind':'value', 'lims':clim,},
        spacing='ico3',
        smoothing_steps=10,
        show_traces=False,
        background="white",
        size=(2000,800)
        # surface="flat",
        # views='flat'
    )
    for label in roi_list:
        brain.add_label(label, alpha=1, borders=True, color='white')
    
    if view == 1:
        brain.show_view(col = 1, view={'elevation':90, 'azimuth':35})
        brain.show_view(col = 0, view={'elevation':90, 'azimuth':145})
    elif view == 2:
        brain.show_view(col = 1, view={'elevation':170})
        brain.show_view(col = 0, view={'elevation':170})

# chroma window
vis_brain(stc_feat, t_min=0.3, t_max=0.32, clim=[1.9,2.0,2.1], view=1)
vis_brain(stc_feat, t_min=0.3, t_max=0.32, clim=[1.9,2.0,2.1], view=2)

# height window
vis_brain(stc_feat, t_min=0.18, t_max=0.23, clim=[1.8,1.9,2], view=1)
vis_brain(stc_feat, t_min=0.18, t_max=0.23, clim=[1.8,1.9,2], view=2)


# %% weight time series

import seaborn as sns

roi_coef_time=[]
roi_name_list=[]
for label in roi_list:
    roi_coef_time.append(stc_feat.in_label(label).data.mean(axis=0))
    roi_name_list.append(label.name)
roi_name_list = [item.replace('-lh', '-L').replace('-rh', '-R') for item in roi_name_list]

    
roi_coef_time = np.stack(roi_coef_time)
column_sums = np.sum(roi_coef_time, axis=0)
roi_coef_time_norm = roi_coef_time / column_sums
# roi_coef_time_norm = roi_coef_time

df = pd.DataFrame(
    {'data': np.concatenate([x for x in roi_coef_time_norm]),
     'time': np.tile(timeAxis, 12),
     'roi_name': [item for item in roi_name_list for _ in range(roi_coef_time_norm.shape[1])]
     }
     )

df['Region'] = df['roi_name'].str[:-2]
df['Hemisphere'] = df['roi_name'].str[-1:]

with plt.style.context('seaborn-notebook'):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    colors = pl.cm.tab10(np.linspace(0,1,6))
    
    fig, ax = plt.subplots(1,1, figsize=(7, 4))

    p = sns.lineplot(data=df, x='time', y='data', hue='Region', style='Hemisphere', ax=ax)
    # p.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    line_chance = ax.axhline(1/12, color='k', linestyle='-',zorder=5, lw=0.75)

    
    ax.set_yticks([0.08,0.085,0.09, 1/12])
    ax.set_yticklabels(['8.0', '8.5', '9.0', r'$BL$'], fontsize=8)
    ax.set_ylabel('proportional absolute weight (%)', fontsize=8)
    ax.set_xticks([-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
    ax.set_xlabel('time (s)', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlim(-0.2, 0.5)
    
    ax.legend(fontsize=7, framealpha=0)

    fig.tight_layout()
    
    plt.show()

    plt.savefig('fig/weight_time.png', format='png', dpi=600)

# %%  violin plots

roi_coef_ind=[]

for n in range(len(subCode)):
    stc_ind = mne.SourceEstimate(
        np.abs(patterns_pitch_select_array_ind[n,:,:]),
        vertices=vertices,
        tmin=-0.2,
        tstep=0.005,
        subject="fsaverage",
    )
    roi_coef_time=[]
    roi_name_list=[]
    for label in roi_list:
        roi_coef_time.append(stc_ind.in_label(label).data.mean(axis=0))
        roi_name_list.append(label.name)
    roi_coef_ind.append(np.stack(roi_coef_time))

roi_coef_ind = np.stack(roi_coef_ind)

column_sums = np.sum(roi_coef_ind, axis=1)
roi_coef_ind_norm = roi_coef_ind / np.repeat(column_sums[:, np.newaxis, :], 12, axis=1)





def coef_df(roi_coef_ind_norm, timeAxis, t1, t2):
    t1_i = np.where(timeAxis==t1)[0][0]
    t2_i = np.where(timeAxis==t2)[0][0]
    temp_roi = roi_coef_ind_norm[:,:,t1_i:t2_i].mean(axis=2)
    
    temp_roi_df = pd.DataFrame(temp_roi, columns=roi_name_list)
    
    temp_roi_df.reset_index(inplace=True)
    temp_roi_df.rename(columns={'index':'subj'},inplace=True)
    temp_roi_df['subj'] = temp_roi_df['subj'].astype("category")
    temp_roi_df_melt = pd.melt(temp_roi_df, 'subj')
    
    
    temp_roi_df_melt['Hemisphere']=temp_roi_df_melt['variable'].str[-2:]
    temp_roi_df_melt['Cortex'] = temp_roi_df_melt['variable'].str[:-10]
    
    temp_roi_df_melt.replace({'lh': 'L', 'rh': 'R'}, inplace=True)
    temp_roi_df_melt.drop(columns=['variable'], inplace=True)
    temp_roi_df_melt['value'] = temp_roi_df_melt['value']*100
    return temp_roi_df_melt

roi_df_melt_baseline = coef_df(roi_coef_ind_norm, timeAxis, -0.2, 0)
roi_df_melt_1 = coef_df(roi_coef_ind_norm, timeAxis, 0.18, 0.23)
roi_df_melt_2 = coef_df(roi_coef_ind_norm, timeAxis, 0.3, 0.32)
roi_df_melt_baseline['Time']='baseline'
roi_df_melt_1['Time']='height'
roi_df_melt_2['Time']='chroma'

roi_df_melt_1mB = roi_df_melt_1.copy()
roi_df_melt_1mB['value'] = roi_df_melt_1['value'] - roi_df_melt_baseline['value']

roi_df_melt_2mB = roi_df_melt_2.copy()
roi_df_melt_2mB['value'] = roi_df_melt_2['value'] - roi_df_melt_baseline['value']

roi_df_melt_diff = roi_df_melt_1.copy().drop(columns='Time')
roi_df_melt_diff['value']=roi_df_melt_2['value']-roi_df_melt_1['value']

roi_df_melt = pd.concat([roi_df_melt_1,roi_df_melt_2],ignore_index=True)

with plt.style.context('seaborn-notebook'):
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    fig, ax = plt.subplots(2,3, figsize=(8, 4))
    label_list = ['Early Auditory', 'Auditory Association','Orbital and Polar Frontal','Insular and Frontal Opercular','Inferior Frontal','DorsoLateral Prefrontal']
    for c in range(len(label_list)):
        temp_label=label_list[c]
        temp_ax=ax[c//3,c%3]
        custom_palette = {'height': 'cyan', 'chroma': 'pink'}
        sns.violinplot(data=roi_df_melt[roi_df_melt['Cortex']==temp_label], x="Hemisphere", y="value", hue="Time", inner='point', ax=temp_ax, width=0.5, linewidth=1, palette=custom_palette, cut=0)
        temp_ax.tick_params(labelsize=6)   
        temp_ax.legend(fontsize=6.5, framealpha=0,loc='upper center')
        temp_ax.set_ylabel('proportional absolute weight (%)',fontsize=6)
        temp_ax.set_xlabel('')
        temp_ax.axhline(100/12, color='k', linestyle=':',zorder=0, lw=1)
        temp_ax.set_title(temp_label,fontsize=7)
        
    fig.tight_layout()
    plt.show()
    plt.savefig('fig/weight_violins.png', format='png', dpi=600)



# %% testing weights

            
print('*** chroma - height ***')
for c in label_list:
    for h in ['L','R']:
        temp_df = roi_df_melt[(roi_df_melt['Cortex']==c) &(roi_df_melt['Hemisphere']==h)]
        x = np.array(temp_df[temp_df['Time']=='chroma']['value'])
        y = np.array(temp_df[temp_df['Time']=='height']['value'])
        result=stats.ttest_rel(x,y)
        if result.pvalue<0.01:
            print(c+'-'+h+' p-value: '+str(result.pvalue))
            print(c+'-'+h+' statistics: '+str(result.statistic))
            print(c+'-'+h+' Cohen D: '+str((np.mean(x)-np.mean(y))/np.std(x-y)))

print('*** Right - Left ***')
for c in label_list:
    for t in ['height','chroma']:
        temp_df = roi_df_melt[(roi_df_melt['Cortex']==c) &(roi_df_melt['Time']==t)]
        x = np.array(temp_df[temp_df['Hemisphere']=='R']['value'])
        y = np.array(temp_df[temp_df['Hemisphere']=='L']['value'])
        result=stats.ttest_rel(x,y)
        if result.pvalue<0.01:
            print(c+'-'+t+' p-value: '+str(result.pvalue))
            print(c+'-'+t+' statistics: '+str(result.statistic))
            print(c+'-'+t+' Cohen D: '+str((np.mean(x)-np.mean(y))/np.std(x-y)))
            
print('*** Above chance 100/12 ***')
for c in label_list:
    for h in ['L','R']:
        for t in  ['height','chroma']:
            temp_df = roi_df_melt[(roi_df_melt['Cortex']==c) & (roi_df_melt['Hemisphere']==h) & (roi_df_melt['Time']==t)]
            x=np.array(temp_df['value'])-100/12
            result=stats.ttest_1samp(x,0)
            if (result.pvalue<0.05) & (result.statistic>0):
                print(c+'-'+h+'-'+t+' p-value: '+str(result.pvalue)) 
                print(c+'-'+h+'-'+t+' statistics: '+str(result.statistic))
                print(c+'-'+h+'-'+t+' Cohen D: '+str(np.mean(x)/np.std(x)))
                
                
                