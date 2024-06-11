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


scores_pitch_all = []
score_tempGen_pitch_all = []

for subject in subCode:

    
    fband = 'delta' # change this into 'theta', 'alpha', 'beta', and 'gamma'
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

# %% cluter permutation test on the overall pitch decoding accuracy



width_threshold = 1
threshold=t.isf(0.05/2,len(subCode)-1)
cluster_threshold = 0.01
exclude=timeAxis<0

score_subj = np.nanmean(scores_pitch_all, axis = (1,3,4))
t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(X=score_subj-0.5, threshold=threshold, exclude=exclude, tail=0, n_permutations=5000, n_jobs=-1, seed=23)


fig, ax = plt.subplots(1, figsize=(5, 4))
# fig.suptitle('Pitch decoding AUC (averaged across pitch pairs)')
plt.style.use('seaborn-notebook')
line_subj = ax.plot(timeAxis, score_subj.T, color='k', alpha = 0.2, label='participant',zorder=2)
line_mean = ax.plot(timeAxis, score_subj.mean(axis=0), color='m', linewidth=3, label='mean',zorder=10)
line_chance = ax.axhline(.5, color='k', linestyle='--', label='chance',zorder=1)
ax.axvline(0, color='k')
# area_abovechance = ax.axvspan(xmin=t_min, xmax=t_max, ymin=0, ymax=1, color='b', alpha=0.15)
plt.xlim([-0.2,0.5])
for n in range(len(clusters)):
    if (cluster_pv[n]<cluster_threshold) & (len(clusters[n][0])>=width_threshold):
        t_min = timeAxis[clusters[n][0][0]]
        t_max = timeAxis[clusters[n][0][-1]]
        area_abovechance = ax.axvspan(xmin=t_min, xmax=t_max, ymin=0, ymax=1, color='b', alpha=0.15)
        text_pos = [(t_min+t_max)/2-0.01, 0.56]
        rotation=0
        # if t_max-t_min < 0.1:
        #     rotation = 90
        #     text_pos[1] -= 0.005
        # if cluster_pv[n] < 0.002:
        #     ax.text(text_pos[0], text_pos[1], 'p < 0.001', fontstyle='italic', fontsize=8, rotation=rotation)
        # else:
        #     ax.text(text_pos[0], text_pos[1], 'p = '+str(round(cluster_pv[n],3)), fontstyle='italic', fontsize=8, rotation=rotation)
ax.legend([line_subj[0], line_mean[0], area_abovechance],['participant mean','grand mean', 'p < 0.01'], loc='best', fontsize=6)

# if fband=='delta':
#     text_pos = [0.20, 0.56]
#     ax.text(text_pos[0], text_pos[1], 'p = '+str(round(cluster_pv[clus_ind],3)), fontstyle='italic',fontsize=8)
# elif fband=='theta':
#     text_pos = [0.12, 0.47]
#     ax.text(text_pos[0], text_pos[1], 'p < 0.001', fontstyle='italic',fontsize=8)
ax.set_title(fband, fontsize=10)
ax.set_xlabel('time (s)', fontsize=10)
ax.set_ylabel('ROC-AUC', fontsize=10)
ax.tick_params(labelsize=8)
# plt.savefig(file_folder+'/permutation_MLM_time_sepModels_reFull_spearmanCoch/figs/pitch_auc.tif')

# %% plot by-pair AUC time series



score_pair=[]
dist_list = []
for n1 in range(8):
    for n2 in range(n1+1,8):
        score_pair.append(np.nanmean(scores_pitch_all[:,:,:,n1,n2], axis = (0,1)))
        dist_list.append((n2-n1)/3)

score_pair = np.vstack(score_pair)
dist_list = np.hstack(dist_list)

width_threshold = 1
threshold=t.isf(0.05/2,len(score_pair)-1)
cluster_threshold = 0.01
exclude=timeAxis<0

segs = [np.column_stack([timeAxis, y]) for y in score_pair]
        
t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(X=score_pair-0.5, threshold=threshold, exclude=exclude, tail=0, n_permutations=5000, n_jobs=-1, seed=23)




cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=1, vmax=7)

fig, ax = plt.subplots(1, figsize=(5, 4))
# fig.suptitle('Pitch decoding AUC (averaged across pitch pairs)')
plt.style.use('seaborn-notebook')

line_segments = LineCollection(segs, array=dist_list, alpha=0.5)
ax.add_collection(line_segments)

axcb = fig.colorbar(line_segments)
axcb.set_label('pitch height difference (# of octaves)', fontsize=10)
axcb.set_ticks([1/3, 2/3, 1, 4/3, 5/3, 2, 7/3])
axcb.set_ticklabels(['1/3', '2/3', '1', '4/3', '5/3', '2', '7/3'])
axcb.ax.tick_params(labelsize=8)


line_mean = ax.plot(timeAxis, score_pair.mean(axis=0), color='m', linewidth=3, label='mean',zorder=10)
line_chance = ax.axhline(.5, color='k', linestyle='--', label='chance',zorder=5)
ax.axvline(0, color='k')
for n in range(len(clusters)):
    if (cluster_pv[n]<cluster_threshold) & (len(clusters[n][0])>=width_threshold):
        t_min = timeAxis[clusters[n][0][0]]
        t_max = timeAxis[clusters[n][0][-1]]
        area_abovechance = ax.axvspan(xmin=t_min, xmax=t_max, ymin=0, ymax=1, color='b', alpha=0.15)
        text_pos = [(t_min+t_max)/2-0.01, 0.55]
        rotation=0
        # if t_max-t_min < 0.1:
        #     rotation = 90
        #     text_pos[1] -= 0.005
        # if cluster_pv[n] < 0.002:
        #     ax.text(text_pos[0], text_pos[1], 'p < 0.001', fontstyle='italic', fontsize=8, rotation=rotation)
        # else:
        #     ax.text(text_pos[0], text_pos[1], 'p = '+str(round(cluster_pv[n],3)), fontstyle='italic', fontsize=8, rotation=rotation)
plt.xlim([-0.2,0.5])
ax.legend([line_mean[0], area_abovechance],['grand mean', 'p < 0.01'], loc='best', fontsize=6)
# if fband=='delta':
#     text_pos = [0.20, 0.56]
#     ax.text(text_pos[0], text_pos[1], 'p = '+str(round(cluster_pv[clus_ind],3)), fontstyle='italic',fontsize=8)
# elif fband=='theta':
#     text_pos = [0.12, 0.47]
#     ax.text(text_pos[0], text_pos[1], 'p < 0.001', fontstyle='italic',fontsize=8)

ax.set_title(fband, fontsize=10)
ax.set_xlabel('time (s)', fontsize=10)
ax.set_ylabel('ROC-AUC', fontsize=10)
ax.tick_params(labelsize=8)



# %% write a bootstrapping method
# read the files produced by decode02_HPC_pitchMLM_permutation.py



   
def cal_clu_stats(data, threshold):
    from scipy.ndimage import label
    import numpy as np
    mask_tt = np.abs(data)>=threshold
    labeled_array, num_features = label(mask_tt)
    temp_clus = []
    for mask_n in range(1,num_features+1):
        temp_clus.append(np.sum(data[labeled_array==mask_n]))
    return temp_clus, labeled_array



def plot_mlm_time(col_name_list, df_tvalues, dir_list, threshold, t_min, t_max, model_name):
    plt.style.use('seaborn-notebook')
    fig, ax = plt.subplots(3,1, figsize=(4, 5))
    fig.tight_layout()
    ax_n=0
    for col_name in col_name_list:
        clus_stats_perm = []
        for nFile in dir_list:
            df = pd.read_csv(nFile, index_col=(0)) 
            data = df[col_name]
            data = data[(data.index>=t_min) & (data.index<=t_max)]
            temp_clus_stats, _ = cal_clu_stats(data, threshold)
            try:
                max_clus_stats = max(temp_clus_stats, key=abs)
            except:
                max_clus_stats=0
            clus_stats_perm.append(max_clus_stats)
            
        data_orig = df_tvalues[col_name]
        data_orig = data_orig[(data_orig.index>=t_min) & (data_orig.index<=t_max)]
        clus_stats_orig_list, labeled_array = cal_clu_stats(data_orig, threshold)
        
        p_val_list = []
        for clus_stats_orig in clus_stats_orig_list:
            if clus_stats_orig > 0:
                p_val = np.mean(np.array(clus_stats_perm)>clus_stats_orig)*2
            elif clus_stats_orig <= 0:
                p_val = np.mean(np.array(clus_stats_perm)<clus_stats_orig)*2
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
        if ax_n==2:
            ax[ax_n].set_xlabel('time (s)', fontsize=8)
            ax[ax_n].set_ylabel('t-value', fontsize=8)
        else:
            plt.setp(ax[ax_n].get_xticklabels(), visible=False)
            
        
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
                if p_val_list[n]==0:
                    p_val_text='p < 0.001'
                else:
                    p_val_text='p = '+str(round(p_val_list[n],3))
                ax[ax_n].text(text_position[0], text_position[1], p_val_text, fontstyle='italic', rotation = 90, fontsize=4.5)
        
        ax_n+=1
        
        # line_thresh = ax.axhline(y = area_thresh, color = '0.8', linestyle = '--', label='cluster threshold')
        # ax.legend([area_clus], ['significant cluster'])
        
    plt.show()


# 'samePitch': chroma equivalence
# 'coch': cochleagram similarity
model_sel = 'samePitch' 

if model_sel=='samePitch':
    dir_list = glob.glob(file_folder+'permutation_MLM_time_sepModels_reFull_spearmanCoch_17subjs/'+fband+ '*samePitch*perm*.csv')
    dir_orig = glob.glob(file_folder+'permutation_MLM_time_sepModels_reFull_spearmanCoch_17subjs/'+fband+ '*samePitch*orig*.csv')
    col_name_list = ['pitchDist','samePitch[T.True]','pitchDist:samePitch[T.True]']
    model_name=''
elif model_sel=='coch':
    dir_list = glob.glob(file_folder+'permutation_MLM_time_sepModels_reFull_spearmanCoch_17subjs/'+fband+ '*coch*perm*.csv')
    dir_orig = glob.glob(file_folder+'permutation_MLM_time_sepModels_reFull_spearmanCoch_17subjs/'+fband+ '*coch*orig*.csv')
    col_name_list = ['pitchDist','coch','pitchDist:coch']
    model_name=''

    
df_tvalues = pd.read_csv(dir_orig[0], index_col=(0)) 

# threshold = t.isf(0.05/2,len(subCode)-1)
threshold = 2.5
t_min = 0
t_max = 0.5

plot_mlm_time(col_name_list, df_tvalues, dir_list, threshold, t_min, t_max, model_name)

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
# ax1.set_yticks(y)
# ax1.set_yticklabels(fontsize=6)
ax1.tick_params(labelsize=6)
 
# Show the plot
ax1.tick_params(bottom=True, top=True, left=True, right=True)
ax1.set_xlim(0.25, 2.65)
ax1.set_xlabel('Pitch height difference (# of octaves)',fontsize=8)
ax1.set_ylabel('Neural dissimilarity (ROC-AUC)',fontsize=8)
# legend1=plt.legend(title='model at time (s)', fontsize=8, title_fontsize=6)
ax1.legend(handles=legend_elements, loc='best', fontsize=6, title='Pitch chroma equivalence', title_fontsize=6)
# plt.gca().add_artist(legend1)


pair_count = list(chain.from_iterable([[x1[n]]*(7-n) for n in range(len(x1))]))
# ax2.grid(axis='y',alpha=0.5, ls=':', lw=0.5)
ax2.bar(x1, [7,6,5,4,3,2,1], width=0.2, color='pink')
ax2.set_xticks(x1)
ax2.set_xlim(0.25, 2.65)
ax2.set_yticks([1,3,5,7])
ax2.set_ylabel('# of pitch pairs',fontsize=5)
ax2.tick_params(labelsize=5, bottom=False)
plt.setp(ax2.get_xticklabels(), visible=False)


plt.show()


# %% temporal generalization


tempGen_list = []
pitch_mingap=3
contour_p = 0.01

for n1 in range(score_tempGen_pitch_all.shape[4]-pitch_mingap):
    for n2 in range(n1+pitch_mingap, score_tempGen_pitch_all.shape[5]):
        tempGen_list.append(score_tempGen_pitch_all[:,:,:,:,n1,n2].mean(axis=1))
        
tempGen_mat=np.array(tempGen_list).mean(axis=0)

tempGen_mean = np.nanmean(tempGen_mat, axis=(0))


xx, yy = np.meshgrid(timeAxis, timeAxis)
mask2D = (xx<0) + (yy<0)


result=stats.ttest_1samp(tempGen_mat, 0.5, alternative='greater')

# T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(tempGen_mat-0.5, n_permutations=5000, n_jobs=-1, exclude=mask2D)
# tempGen_sig_plot = np.nan * np.ones_like(T_obs)
# for c, p_val in zip(clusters, cluster_p_values):
#     if p_val <= 0.05:
#         tempGen_sig_plot[c] = tempGen_mean[c]

limit = max(abs(np.min(tempGen_mean)-0.5), abs(np.max(tempGen_mean)-0.5) )

fig, ax = plt.subplots()
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
    np.where(mask2D, np.nan, result.pvalue),
    # result.pvalue * np.logical_not(mask2D),
    # result.statistic>2.5,
    # tempGen_sig_plot>0,
    levels=[0.01],
    origin="lower",
    colors='k',
    linewidths=1,
    extent=[timeAxis[0], timeAxis[-1], timeAxis[0], timeAxis[-1]],
    )
ax.axhline(0.0, color="k",lw=1,ls=':')
ax.axvline(0.0, color="k",lw=1,ls=':')
ax.xaxis.set_ticks_position("bottom")
ax.set_xlabel('Testing time (s)')
ax.set_ylabel('Training time (s)')
h,l = contour.legend_elements('p')
ax.legend(h,l,loc='lower right')
# ax.set_title("Generalization across time and condition", fontweight="bold")
fig.colorbar(im, ax=ax, label="Mean performance (ROC-AUC)")
plt.show()

# %% multidimemsional scaling

from sklearn.manifold import MDS


def plot_mds(distances, t_model):

    distances = np.insert(distances, 0, 0) - 0.495  # Insert '0' at index 0, and then substract 0.495
    
    distance_matrix = np.zeros((8,8,))
    
    for n1 in range(8):
        for n2 in range(8):
            if n1 != n2:
                distance_matrix[n1, n2] = distances[abs(n1-n2)]

    # Create an MDS object for 2D scaling
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=91) # random_state=91 is good!
    
    # Fit the model and transform the distance matrix into 3D coordinates
    coordinates = mds.fit_transform(distance_matrix)
    
    
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    
    # Create a 2D scatter plot
    plt.style.use('seaborn-notebook')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    
    # Unpack coordinates for easier plotting
    x, y = coordinates[:, 0], coordinates[:, 1]
    
    # Plot the points
    ax.scatter(x, y, c=[1,2,3,1,2,3,1,2])
    
    # Annotate points with their index
    for i, p_name in zip(range(len(x)), ["G#6", "C7", "E7", "G#7", "C8", "E8", "G#8", "C9"]):
        ax.text(x[i], y[i], p_name)
    
    # Set labels for axes
    ax.set_xlim([-0.03,0.03])
    ax.set_ylim([-0.03,0.03])
    ax.set_title("Time: " +str(t_model)+" (s)")
    
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
    if 0.3 <= t_model <=0.32:
        for n in range(len(x1)):
            y.append(coefs['Intercept'] + coefs['pitchDist']*x1[n] + coefs['samePitch[T.True]']*x2[n] + coefs['pitchDist:samePitch[T.True]']*x1[n]*x2[n])
    else:
        for n in range(len(x1)):
            y.append(coefs['Intercept'] + coefs['pitchDist']*x1[n])
    
    distances = y
    plot_mds(distances, t_model)
    plt.savefig('time'+str(int(t_model*1000))+'ms.png', format='png', dpi=600)

plt.close('all')
# %% 3D multidimemsional scaling

import matplotlib.pyplot as plt
import numpy as np

def plot_manual_3Dmds(distances, t_model):

    if 0.3 <= t_model <=0.32:
        distances = np.insert(distances, 0, 0) - 0.495 # Insert '0' at index 0, and then substract 0.495
    else:
        distances = np.insert(distances, 0, 0) - min(distances)  # Insert '0' at index 0, and then substract 0.495
      
    distance_matrix = np.zeros((8,8,))
    
    for n1 in range(8):
        for n2 in range(8):
            if n1 != n2:
                distance_matrix[n1, n2] = distances[abs(n1-n2)]

    if 0.3 <= t_model <=0.32:
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=6) # random_state=6 is good!
        coordinates = mds.fit_transform(distance_matrix)

        # Convert degrees to radians
        # azim = np.deg2rad(-40)
        azim = np.deg2rad(-275)
        elev = np.deg2rad(-265)

        # Define the rotation matrix for rotation around the z-axis (azimuth)
        R_z = np.array([[np.cos(azim), -np.sin(azim), 0],
                        [np.sin(azim), np.cos(azim), 0],
                        [0, 0, 1]])

        # Define the rotation matrix for rotation around the x-axis (elevation)
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(elev), -np.sin(elev)],
                        [0, np.sin(elev), np.cos(elev)]])

        # Combine the rotations: Note that R_x is applied first, then R_z
        R_combined = np.dot(R_z, R_x)

        # Apply the combined rotation to the coordinates
        rotated_coordinates = np.dot(coordinates, R_combined.T)

        x = rotated_coordinates[:,0]
        y = rotated_coordinates[:,1]
        z = rotated_coordinates[:,2]
        z -= min(z)
        
    else:
        z = distance_matrix[:,0]
        x = np.zeros_like(z)
        y = np.zeros_like(z)
    
    
    import matplotlib.pyplot as plt
    
    
    # Create a 3D scatter plot
    plt.style.use('seaborn-notebook')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x, y, z, c=[1,2,3,1,2,3,1,2], label='Points')

    # Annotate points with their index
    for i, p_name in zip(range(len(x)), ["G#6", "C7", "E7", "G#7", "C8", "E8", "G#8", "C9"]):
        ax.text(x[i], y[i], z[i], p_name)

    # Connect the points with a line
    ax.plot(x, y, z, color='r', label='Line')

    # Add labels
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    ax.set_xlim([-0.01,0.01])
    ax.set_ylim([-0.01,0.01])
    ax.set_zlim([0,0.05])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title("Time: " +str(t_model)+" (s)")


    # # Show the legend
    # ax.legend()

    # ax.view_init(10, -40)
    ax.view_init(10, -55)


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
    if 0.3 <= t_model <=0.32:
        for n in range(len(x1)):
            y.append(coefs['Intercept'] + coefs['pitchDist']*x1[n] + coefs['samePitch[T.True]']*x2[n] + coefs['pitchDist:samePitch[T.True]']*x1[n]*x2[n])
    else:
        for n in range(len(x1)):
            y.append(coefs['Intercept'] + coefs['pitchDist']*x1[n])
    
    distances = y
    plot_manual_3Dmds(distances, t_model)
    plt.savefig('3D_time'+str(int(t_model*1000))+'ms.png', format='png', dpi=600)

plt.close('all')

