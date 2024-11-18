#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:58:51 2022

@author: andrewchang
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import glob
from itertools import chain
from mne.stats import permutation_cluster_1samp_test
from scipy import stats
from scipy.stats import t
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec


plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


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


file_folder = 'sourceSTC20230711_ico3_freqBands_shuffled/decodeSource20230711_RidgeCV/auditory_frontal_alpha10^(-2)-10^3_41grid_correctPitchCoefPattern/'
# perm_folder = 'permutation_MLM_time_sepModels_reFull_spearmanCoch_17subjs/'
perm_folder = 'permutation_MLM_time_sepModels_reFull_spearmanCoch_17subjs_NC_10k/'

save_fig = True


def load_data(subCode, file_folder, fband):
    # fband = 'delta', 'theta', 'alpha', 'beta', and 'gamma'
    
    scores_pitch_all = []
    score_tempGen_pitch_all = []
    
    for subject in subCode:
    
        with open(file_folder+subject+'_'+fband+'_decode', 'rb') as fp:
            _, _, _, score_pitch, _, _, _, patterns_pitch = pickle.load(fp)  
            fp.close()
            
        with open(file_folder+subject+'_'+fband+'_decode_tempGen', 'rb') as fp:
            score_tempGen_pitch = pickle.load(fp)  
            fp.close()
        
        scores_pitch_all.append(score_pitch)
        score_tempGen_pitch_all.append(score_tempGen_pitch)
        
        del score_pitch, patterns_pitch, score_tempGen_pitch
    
    scores_pitch_all = np.stack([scores_pitch_all[i] for i in subSel], axis=0)
    score_tempGen_pitch_all = np.stack([score_tempGen_pitch_all[i] for i in subSel], axis=0)
    
    return scores_pitch_all, score_tempGen_pitch_all

# %% cluter permutation test on the overall pitch decoding accuracy

fband = 'beta'
scores_pitch_all, _ = load_data(subCode, file_folder, fband)

width_threshold = 5
# threshold=t.isf(0.05/2,len(score_pair)-1)
threshold=2.5
cluster_threshold = 0.01
exclude=timeAxis<0
n_permutations=10000

score_subj = np.nanmean(scores_pitch_all, axis = (1,3,4))
t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(X=score_subj-0.5, threshold=threshold, exclude=exclude, tail=0, n_permutations=n_permutations, n_jobs=-1, seed=23)


fig, ax = plt.subplots(1, figsize=(6, 3.5))
plt.style.use('seaborn-paper')
line_subj = ax.plot(timeAxis, score_subj.T, color='k', alpha = 0.2, label='participant',zorder=2)
line_mean = ax.plot(timeAxis, score_subj.mean(axis=0), color='r', linewidth=3, label='mean',zorder=10)
line_chance = ax.axhline(.5, color='k', linestyle='--', linewidth=1, label='chance',zorder=1)
ax.axvline(0, color='k', linewidth=1)
# area_abovechance = ax.axvspan(xmin=t_min, xmax=t_max, ymin=0, ymax=1, color='b', alpha=0.15)
plt.xlim([-0.2,0.5])
for n in range(len(clusters)):
    if (cluster_pv[n]<cluster_threshold) & (len(clusters[n][0])>=width_threshold):
        t_min = timeAxis[clusters[n][0][0]]
        t_max = timeAxis[clusters[n][0][-1]]
        area_abovechance = ax.axvspan(xmin=t_min, xmax=t_max, ymin=0, ymax=1, color='b', alpha=0.15)
        text_pos = [(t_min+t_max)/2-0.01, 0.56]
        rotation=0
ax.legend([line_subj[0], line_mean[0], area_abovechance],['participant mean','grand mean', 'p < 0.01'], loc='best', fontsize=8)

if fband == 'gamma':
    fband = 'low '+fband
ax.set_title(fband, fontsize=20)
ax.set_xlabel('time (s)', fontsize=8)
ax.set_ylabel('ROC-AUC', fontsize=8)
ax.tick_params(labelsize=6)


if save_fig:
    plt.savefig('fig/bySubject_'+fband+'.png', format='png', dpi=600)

# %% plot by-pair AUC time series

fband = 'gamma'
scores_pitch_all, _ = load_data(subCode, file_folder, fband)

score_pair=[]
dist_list = []
for n1 in range(8):
    for n2 in range(n1+1,8):
        score_pair.append(np.nanmean(scores_pitch_all[:,:,:,n1,n2], axis = (0,1)))
        dist_list.append((n2-n1)/3)

score_pair = np.vstack(score_pair)
dist_list = np.hstack(dist_list)

width_threshold = 5
# threshold=t.isf(0.05/2,len(score_pair)-1)
threshold=2.5
cluster_threshold = 0.01
exclude=timeAxis<0
n_permutations=10000

segs = [np.column_stack([timeAxis, y]) for y in score_pair]
        
t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(X=score_pair-0.5, threshold=threshold, exclude=exclude, tail=0, n_permutations=n_permutations, n_jobs=-1, seed=23)




cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=1, vmax=7)

fig, ax = plt.subplots(1, figsize=(6, 3.5))
# fig.suptitle('Pitch decoding AUC (averaged across pitch pairs)')
plt.style.use('seaborn-paper')

line_segments = LineCollection(segs, array=dist_list, alpha=0.25)
ax.add_collection(line_segments)

show_colorbar = False

if show_colorbar:
    axcb = fig.colorbar(line_segments)
    axcb.set_label('pitch height difference (# of octaves)', fontsize=8)
    axcb.set_ticks([1/3, 2/3, 1, 4/3, 5/3, 2, 7/3])
    axcb.set_ticklabels(['1/3', '2/3', '1', '4/3', '5/3', '2', '7/3'])
    axcb.ax.tick_params(labelsize=6)


line_mean = ax.plot(timeAxis, score_pair.mean(axis=0), color='r', linewidth=3, label='mean',zorder=10)
line_chance = ax.axhline(.5, color='k', linestyle='--', label='chance',zorder=5, linewidth=1)
ax.axvline(0, color='k', linewidth=1)
for n in range(len(clusters)):
    if (cluster_pv[n]<cluster_threshold) & (len(clusters[n][0])>=width_threshold):
        t_min = timeAxis[clusters[n][0][0]]
        t_max = timeAxis[clusters[n][0][-1]]
        area_abovechance = ax.axvspan(xmin=t_min, xmax=t_max, ymin=0, ymax=1, color='b', alpha=0.15)
        text_pos = [(t_min+t_max)/2-0.01, 0.55]
        rotation=0

plt.xlim([-0.2,0.5])
ax.legend([line_mean[0], area_abovechance],['grand mean', 'p < 0.01'], loc='best', fontsize=8)

if fband == 'gamma':
    fband = 'low '+fband
ax.set_title(fband, fontsize=20)
ax.set_xlabel('time (s)', fontsize=8)
ax.set_ylabel('ROC-AUC', fontsize=8)
ax.tick_params(labelsize=6)

if save_fig:
    plt.savefig('fig/byItem_'+fband+'.png', format='png', dpi=600)


# %% write a bootstrapping method
# read the files produced by decode02_HPC_pitchMLM_permutation.py

   
# def cal_clu_stats(data, threshold):
#     from scipy.ndimage import label
#     import numpy as np
#     mask_tt = np.abs(data)>=threshold
#     labeled_array, num_features = label(mask_tt)
#     temp_clus = []
#     for mask_n in range(1,num_features+1):
#         temp_clus.append(np.sum(data[labeled_array==mask_n]))
#     return temp_clus, labeled_array

def cal_clu_stats(data, threshold, width_threshold):
    from scipy.ndimage import label
    import numpy as np
    mask_tt = np.abs(data)>=threshold
    labeled_array, num_features = label(mask_tt)
    temp_clus = []
    for mask_n in range(1,num_features+1):
        if sum(labeled_array==mask_n) >= width_threshold:
            temp_clus.append(np.sum(data[labeled_array==mask_n]))
        else: 
            temp_clus.append(0)
    return temp_clus, labeled_array

def plot_nc(nc_df, t_min, t_max, ax_n):
    nc_df = nc_df[(nc_df.index>=t_min) & (nc_df.index<=t_max)]
    if 'samePitch[T.True]' in col_name_list:
        y1 = nc_df['r_samePitch_lowbound_time']
        y2 = nc_df['r_samePitch_highbound_time']
    elif 'coch' in col_name_list:
        y1 = nc_df['r_coch_lowbound_time']
        y2 = nc_df['r_coch_highbound_time']
    
    # make plot
    ax[ax_n].set_title('noise ceiling', fontdict=({'size':8, 'style':'oblique'}), loc='left')
    ax[ax_n].spines['bottom'].set_position(('data', 0))
    ax[ax_n].spines['top'].set_visible(False)
    ax[ax_n].spines['right'].set_visible(False)
    ax[ax_n].grid(visible=True, axis='y', linestyle=':', linewidth=0.5, which='major')
    ax[ax_n].fill_between(nc_df.index, y1, y2, color='tab:gray')
    ax[ax_n].tick_params(labelsize=6)
    ax[ax_n].set_xticks([0.1,0.2,0.3,0.4,0.5])
    ax[ax_n].set_xlim(0,0.5)
    ax[ax_n].set_yticks([0.15,0.3])
    ax[ax_n].set_ylabel("Pearson's r", fontsize=8)
    ax[ax_n].set_xlabel('time (s)', fontsize=8)

def plot_mlm_time(col_name_list, df_tvalues, dir_list, threshold, width_threshold, t_min, t_max, model_name, samePitch_r2_df, coch_r2_df):
    plt.style.use('seaborn-notebook')
    fig, ax = plt.subplots(4,1, figsize=(3.5, 5), gridspec_kw={'height_ratios': [2, 2, 2, 0.8]})
    ax_n=0
    for col_name in col_name_list:
        clus_stats_perm = []
        for nFile in dir_list:
            df = pd.read_csv(nFile, index_col=(0)) 
            data = df[col_name]
            data = data[(data.index>=t_min) & (data.index<=t_max)]
            temp_clus_stats, _ = cal_clu_stats(data, threshold, width_threshold)
            try:
                max_clus_stats = max(temp_clus_stats, key=abs)
            except:
                max_clus_stats=0
            clus_stats_perm.append(max_clus_stats)
            
        data_orig = df_tvalues[col_name]
        data_orig = data_orig[(data_orig.index>=t_min) & (data_orig.index<=t_max)]
        clus_stats_orig_list, labeled_array = cal_clu_stats(data_orig, threshold, width_threshold)
        
        p_val_list = []
        for clus_stats_orig in clus_stats_orig_list:
            if clus_stats_orig > 0:
                p_val = np.mean(np.array(clus_stats_perm)>clus_stats_orig)*2
            elif clus_stats_orig < 0:
                p_val = np.mean(np.array(clus_stats_perm)<clus_stats_orig)*2
            else:
                p_val = 1
            print('permutation p-value (2-tailed): '+str(p_val))
            p_val_list.append(p_val)
        
        if col_name=='pitchDist':
            plot_title = model_name+'Height difference'
            line_color = 'blue'
        elif col_name=='coch': 
            plot_title = model_name+'Cochleagram similarity'
            line_color = 'olive'
        elif col_name=='samePitch[T.True]': 
            plot_title = model_name+'Chroma equivalence'   
            line_color = 'red'
        elif col_name=='pitchDist:samePitch[T.True]': 
            plot_title = model_name+'Height difference * Chroma equivalence'   
            line_color = 'purple'
        elif col_name=='pitchDist:coch': 
            plot_title = model_name+'Height difference * Cochleagram similarity'   
            line_color = 'green'
        else:
            plot_title = model_name+'??'   
            line_color = 'black'
        
        # make plot
        ax[ax_n].set_title(plot_title, fontdict=({'size':8, 'style':'oblique'}), loc='left')
        ax[ax_n].spines['bottom'].set_position(('data', 0))
        ax[ax_n].spines['top'].set_visible(False)
        ax[ax_n].spines['right'].set_visible(False)
        ax[ax_n].plot(data_orig.index, data_orig, color=line_color)
        ax[ax_n].tick_params(labelsize=6)
        ax[ax_n].set_xticks([0.1,0.2,0.3,0.4,0.5])
        ax[ax_n].set_xlim(0,0.5)
        ax[ax_n].set_ylabel('t-value', fontsize=8)
        plt.setp(ax[ax_n].get_xticklabels(), visible=False)
        # if ax_n==2:
        #     ax[ax_n].set_xlabel('time (s)', fontsize=8)
        #     ax[ax_n].set_ylabel('t-value', fontsize=8)
        # else:
        #     plt.setp(ax[ax_n].get_xticklabels(), visible=False)
            
        
        # add area and text of p-value
        for n in range(len(clus_stats_orig_list)):
            clus_stats_orig=clus_stats_orig_list[n]
            if p_val_list[n]<0.05:
                if clus_stats_orig>0:
                    area_ymax, area_ymin = data_orig, 0
                    area_thresh = threshold
                    text_position = [np.mean(data_orig.index[labeled_array==n+1])-0.005, 0.3]
                else:  
                    area_ymax, area_ymin = 0, data_orig
                    area_thresh = -threshold
                    text_position = [np.mean(data_orig.index[labeled_array==n+1])-0.005, -2]
                area_clus = ax[ax_n].fill_between(x=data_orig.index, y1=area_ymax, y2=area_ymin, alpha=0.2, where=labeled_array==n+1, color=line_color)
                if p_val_list[n]<0.001:
                    p_val_text='p < .001'
                else:
                    p_val_text='p = '+"{:.3f}".format(p_val_list[n]).lstrip('0')
                ax[ax_n].text(text_position[0], text_position[1], p_val_text, fontstyle='italic', rotation = 90, fontsize=3.5)
        
        ax_n+=1
        
        # line_thresh = ax.axhline(y = area_thresh, color = '0.8', linestyle = '--', label='cluster threshold')
        # ax.legend([area_clus], ['significant cluster'])
        
        
        if ax_n==3:
            # plot_nc(nc_df, t_min, t_max, ax_n) # plot leave-one-out noise ceiling (not use)
            samePitch_r2_df = samePitch_r2_df[(samePitch_r2_df.index>=t_min) & (samePitch_r2_df.index<=t_max)]
            coch_r2_df = coch_r2_df[(coch_r2_df.index>=t_min) & (coch_r2_df.index<=t_max)]

            if 'samePitch[T.True]' in col_name_list:
                y_m = samePitch_r2_df['R2_m']
                y_c = samePitch_r2_df['R2_c']
            elif 'coch' in col_name_list:
                y_m = coch_r2_df['R2_m']
                y_c = coch_r2_df['R2_c']
            
            # make plot
            # ax[ax_n].set_title('noise ceiling', fontdict=({'size':8, 'style':'oblique'}), loc='left')
            ax[ax_n].spines['bottom'].set_position(('data', 0))
            ax[ax_n].spines['top'].set_visible(False)
            ax[ax_n].spines['right'].set_visible(False)
            # ax[ax_n].grid(visible=True, axis='y', linestyle=':', linewidth=0.5, which='major')
            ax[ax_n].plot(samePitch_r2_df.index, y_m, label = "marginal", color='black', linewidth=1)
            ax[ax_n].plot(samePitch_r2_df.index, y_c, label = "conditional", color='tab:gray', linewidth=1)
            ax[ax_n].legend(fontsize=4, bbox_to_anchor=(0.5, 1.5), loc='upper center', ncol=2)
            # ax[ax_n].legend(fontsize=4, ncol=2)
            ax[ax_n].tick_params(labelsize=6)
            ax[ax_n].set_xticks([0.1,0.2,0.3,0.4,0.5])
            ax[ax_n].set_xlim(0,0.5)
            # ax[ax_n].set_yticks([0.15,0.3])
            ax[ax_n].set_ylabel("$R^2$", fontsize=8)
            ax[ax_n].set_xlabel('time (s)', fontsize=8)
            
    plt.tight_layout()
    plt.show()



# 'samePitch': chroma equivalence
# 'coch': cochleagram similarity
# model_sel = 'samePitch' 
# fband = 'delta'

threshold = 2.5
width_threshold=5

t_min = 0
t_max = 0.5

for model_sel in ['samePitch', 'coch']:
    for fband in ['delta','theta','alpha']:
        if model_sel=='samePitch':
            dir_list = glob.glob(file_folder+perm_folder+fband+ '_samePitch_perm-tval-time*.csv')
            dir_orig = glob.glob(file_folder+perm_folder+fband+ '_samePitch_orig-tval-time.csv')
            col_name_list = ['pitchDist','samePitch[T.True]','pitchDist:samePitch[T.True]']
            model_name=''
        elif model_sel=='coch':
            dir_list = glob.glob(file_folder+perm_folder+fband+ '_coch_perm-tval-time*.csv')
            dir_orig = glob.glob(file_folder+perm_folder+fband+ '_coch_orig-tval-time.csv')
            col_name_list = ['pitchDist','coch','pitchDist:coch']
            model_name=''
        
            
        df_tvalues = pd.read_csv(dir_orig[0], index_col=(0)) 
        
        
        
        nc_df = pd.read_csv(file_folder+perm_folder+fband+ '_noiseCeiling.csv', index_col=0) # not used
        samePitch_r2_df = pd.read_csv(file_folder+perm_folder+fband+ '_samePitch_orig-R2-time.csv', index_col=0)
        coch_r2_df = pd.read_csv(file_folder+perm_folder+fband+ '_coch_orig-R2-time.csv', index_col=0)
        
        plot_mlm_time(col_name_list, df_tvalues, dir_list, threshold, width_threshold, t_min, t_max, model_name, samePitch_r2_df, coch_r2_df)
        
        if save_fig:
            plt.savefig('fig/MLM_'+model_sel+'_'+fband+'_threshold'+str(threshold)+'_widthThreshold'+str(width_threshold)+'.png', format='png', dpi=600)
            plt.close('all')

# %% plot actual MLM prediction


df_fe = pd.read_csv('sourceSTC20230711_ico3_freqBands_shuffled/decodeSource20230711_RidgeCV/auditory_frontal_alpha10^(-2)-10^3_41grid_correctPitchCoefPattern/permutation_MLM_time_sepModels_reFull_spearmanCoch_17subjs/delta_samePitch_orig-fixedeffect-time.csv',
                    index_col=0)

x1 = [1/3, 2/3, 1, 4/3, 5/3, 2, 7/3]
x2 = [0, 0, 1, 0, 0, 1, 0]
t_loc = np.linspace(0, 0.5, 6)
zorder = [0, 0, 0, 1, 0, 0]


fig= plt.figure(figsize = (4,5))
gs = GridSpec(5,1)
ax1 = fig.add_subplot(gs[1:5,:])
ax2 = fig.add_subplot(gs[0,:])

cmap = plt.get_cmap('twilight')
norm = Normalize(vmin=min(t_loc), vmax=max(t_loc)+0.03)

for t_model, z in zip(t_loc, zorder):
    t_model = round(t_model,2)
    y = []
    coefs = df_fe.loc[t_model]
    if (t_model>=0.29) and (t_model<=0.32):
        for n in range(len(x1)):
            y.append(coefs['Intercept'] + coefs['pitchDist']*x1[n] + coefs['samePitch[T.True]']*x2[n] + coefs['pitchDist:samePitch[T.True]']*x1[n]*x2[n])
    else:
        for n in range(len(x1)):
            y.append(coefs['Intercept'] + coefs['pitchDist']*x1[n])
    colors = cmap(norm(t_model))
    
    ax1.plot(x1, y, color=colors, label=str(t_model), zorder=z, linestyle='-', lw=1)
    ax1.scatter(x1, y, facecolors='none', edgecolors=colors, s=20, lw=1)
    ax1.scatter([1,2], [y[i] for i in [2,5]], facecolors=colors, edgecolors=colors, zorder=0, s=20)
    legend_elements = [ax1.scatter([0], [0.5], marker='o', color='k', label='True'),
                       ax1.scatter([0], [0.5], marker='o', color='k', label='False', facecolor='none')]
    ax1.text(x1[-1]+0.07, y[-1], 't='+str(t_model)+'s', fontsize=6, verticalalignment='center', color=colors)
# Create the plot


ax1.set_xticks(x1)
ax1.set_xticklabels(['1/3', '2/3', '1', '4/3', '5/3', '2', '7/3'])
ax1.tick_params(labelsize=6)
 
# Show the plot
ax1.tick_params(bottom=True, top=True, left=True, right=True)
ax1.set_xlim(0.25, 2.65)
ax1.set_xlabel('Pitch height difference (# of octaves)',fontsize=8)
ax1.set_ylabel('Neural dissimilarity (ROC-AUC)',fontsize=8)
ax1.legend(handles=legend_elements, loc='best', fontsize=6, title='Pitch chroma equivalence', title_fontsize=6)


pair_count = list(chain.from_iterable([[x1[n]]*(7-n) for n in range(len(x1))]))
ax2.bar(x1, [7,6,5,4,3,2,1], width=0.2, color='pink')
ax2.set_xticks(x1)
ax2.set_xlim(0.25, 2.65)
ax2.set_yticks([1,3,5,7])
ax2.set_ylabel('# of pitch pairs',fontsize=5)
ax2.tick_params(labelsize=5, bottom=False)
plt.setp(ax2.get_xticklabels(), visible=False)


plt.show()


# %% temporal generalization

fband = 'delta'
scores_pitch_all, score_tempGen_pitch_all = load_data(subCode, file_folder, fband)

tempGen_list = []
pitch_mingap=3

for n1 in range(score_tempGen_pitch_all.shape[4]-pitch_mingap):
    for n2 in range(n1+pitch_mingap, score_tempGen_pitch_all.shape[5]):
        tempGen_list.append(score_tempGen_pitch_all[:,:,:,:,n1,n2].mean(axis=1))
        
tempGen_mat=np.array(tempGen_list).mean(axis=0)

tempGen_mean = np.nanmean(tempGen_mat, axis=(0))


xx, yy = np.meshgrid(timeAxis, timeAxis)
mask2D = (xx<0) + (yy<0)


result=stats.ttest_1samp(tempGen_mat, 0.5, alternative='greater')


limit = max(abs(np.min(tempGen_mean)-0.5), abs(np.max(tempGen_mean)-0.5) )


def plot_tempGen(fdr_bool):
    
    if fdr_bool == False:
        p_val=result.pvalue
        levels=[0.005, 0.01]
        legend_title=""
    elif fdr_bool == True:
        from statsmodels.stats.multitest import fdrcorrection
        _, p_corrected = fdrcorrection(result.pvalue[mask2D==False], alpha=0.05, method='i', is_sorted=False)
        p_val = result.pvalue.copy()
        p_val[-101:,-101:] = p_corrected.reshape(101,101)
        levels=[0.03, 0.04, 0.05]
        legend_title="FDR-corrected"
    
    fig, ax = plt.subplots(figsize = (5,7))
    im = ax.matshow(
        tempGen_mean,
        vmin=0.5-limit,
        vmax=0.5+limit,
        cmap="RdBu_r",
        origin="lower",
        extent=[timeAxis[0], timeAxis[-1], timeAxis[0], timeAxis[-1]],
        aspect='equal'
    )
    contour = ax.contour(
        np.where(mask2D, np.nan, p_val),
        levels=levels,
        origin="lower",
        colors=['k','dimgrey','silver'],
        linewidths=1,
        extent=[timeAxis[0], timeAxis[-1], timeAxis[0], timeAxis[-1]],
        )
    ax.axhline(0.0, color="k",lw=1,ls=':')
    ax.axvline(0.0, color="k",lw=1,ls=':')
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel('Testing time (s)')
    ax.set_ylabel('Training time (s)')
    h,l = contour.legend_elements('p')
    ax.legend(h,l,loc='lower right',title=legend_title,fontsize=8,title_fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean performance (ROC-AUC)")
    plt.tight_layout()
    plt.show()

plot_tempGen(fdr_bool=True)

if save_fig:
    plt.savefig('fig/tempGen_'+fband+'.fit', format='png', dpi=600)

# %% 3D multidimemsional scaling

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.transform import Rotation as R


def plot_manual_3Dmds(distances, t_model):
      
    distances -= min(distances)
    distance_matrix = np.zeros((8,8,))
    
    for n1 in range(8):
        for n2 in range(8):
            if n1 != n2:
                distance_matrix[n1, n2] = distances[abs(n1-n2)]

    if 0.3 <= t_model <=0.32:
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=91)
        coordinates = mds.fit_transform(distance_matrix)
    
        
        
        azimuth = np.deg2rad(0)
        elevation = np.deg2rad(-140)
        rotation = np.deg2rad(10)
        
        r = R.from_euler('zyx', [azimuth, elevation, rotation])
        
        # Apply the rotation to a vector
        rotated_coordinates = r.apply(coordinates)

    
        x = rotated_coordinates[:,0]
        y = rotated_coordinates[:,1]
        z = rotated_coordinates[:,2]
        z -= min(z)
        
    else:
        z = distance_matrix[:,0]
        x = np.zeros_like(z)
        y = np.zeros_like(z)
    
    # rescale the coordinates
    x*=1000
    y*=1000
    z*=1000
    
    # Create a 3D scatter plot
    plt.style.use('seaborn-notebook')
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1.7])

    # Plot the points
    ax.scatter(x, y, z, c=[1,2,3,1,2,3,1,2], label='Points')
    
    # Annotate points with their index
    for i, p_name in zip(range(len(x)), ["G#6", "C7", "E7", "G#7", "C8", "E8", "G#8", "C9"]):
        ax.text(x[i], y[i], z[i], p_name, fontsize=13)
    
    # Connect the points with a line
    ax.plot(x, y, z, color='r', label='Line')
    
    # Add labels

    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([0,60])
    ax.set_xticks([-2,-1,0,1,2])
    ax.set_yticks([-2,-1,0,1,2])
    ax.set_zticks([0,10,20,30,40,50,60])
    ax.set_xticklabels([-2,-1,0,1,2],verticalalignment='baseline')
    ax.set_yticklabels([-2,-1,0,1,2],verticalalignment='baseline',horizontalalignment='left')
    ax.set_zticklabels([0,10,20,30,40,50,60])

    ax.tick_params(axis='both', which='major', labelsize=8, pad=0.1)
    ax.set_title("Time: " +str(t_model)+" (s)", pad=-20)
    
    
    # # Show the legend
    # ax.legend()
    
    # ax.view_init(10, -40)
    ax.view_init(5, 146)
    
    plt.tight_layout()
    # Show the plot
    plt.show()



df_fe = pd.read_csv('sourceSTC20230711_ico3_freqBands_shuffled/decodeSource20230711_RidgeCV/auditory_frontal_alpha10^(-2)-10^3_41grid_correctPitchCoefPattern/permutation_MLM_time_sepModels_reFull_spearmanCoch_17subjs/delta_samePitch_orig-fixedeffect-time.csv',
                    index_col=0)

x1 = [1/3, 2/3, 1, 4/3, 5/3, 2, 7/3]
x2 = [0, 0, 1, 0, 0, 1, 0]
t_loc = np.linspace(0, 0.5, 6)
t_loc = np.linspace(0, 0.5, 51) # for animation use only

for t_model in t_loc:
    t_model = round(t_model,3)
    y = []
    coefs = df_fe.loc[t_model]
    y.append(coefs['Intercept'])
    if 0.3 <= t_model <=0.32:
        for n in range(len(x1)):
            y.append(coefs['Intercept'] + coefs['pitchDist']*x1[n] + coefs['samePitch[T.True]']*x2[n] + coefs['pitchDist:samePitch[T.True]']*x1[n]*x2[n])
    else:
        for n in range(len(x1)):
            y.append(coefs['Intercept'] + coefs['pitchDist']*x1[n])
    
    distances = np.array(y)
    plot_manual_3Dmds(distances, t_model)
    plt.savefig('3D_time'+str(int(t_model*1000))+'ms.png', format='png', dpi=600)

plt.close('all')

