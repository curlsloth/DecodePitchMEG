#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:59:59 2023

@author: andrewchang
adapted from https://github.com/mcdermottLab/pycochleagram/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from pycochleagram import cochleagram as cgram
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# %% cochleagram


def demo_human_cochleagram_helper(signal, sr, n, sample_factor=2, downsample=None, nonlinearity=None):
  """Demo the cochleagram generation.

    signal (array): If a time-domain signal is provided, its
      cochleagram will be generated with some sensible parameters. If this is
      None, a synthesized tone (harmonic stack of the first 40 harmonics) will
      be used.
    sr: (int): If `signal` is not None, this is the sampling rate
      associated with the signal.
    n (int): number of filters to use.
    sample_factor (int): Determines the density (or "overcompleteness") of the
      filterbank. Original MATLAB code supported 1, 2, 4.
    downsample({None, int, callable}, optional): Determines downsampling method to apply.
      If None, no downsampling will be applied. If this is an int, it will be
      interpreted as the upsampling factor in polyphase resampling
      (with `sr` as the downsampling factor). A custom downsampling function can
      be provided as a callable. The callable will be called on the subband
      envelopes.
    nonlinearity({None, 'db', 'power', callable}, optional): Determines
      nonlinearity method to apply. None applies no nonlinearity. 'db' will
      convert output to decibels (truncated at -60). 'power' will apply 3/10
      power compression.

    Returns:
      array:
        **cochleagram**: The cochleagram of the input signal, created with
          largely default parameters.
  """
  human_coch = cgram.human_cochleagram(signal, sr, n=n, sample_factor=sample_factor,
      downsample=downsample, nonlinearity=nonlinearity, strict=False)
  img = np.flipud(human_coch)  # the cochleagram is upside down (i.e., in image coordinates)
  return img

n = 128

df_k = pd.DataFrame()
df_s = pd.DataFrame()
df_f = pd.DataFrame()

coch_k_list = []
coch_s_list = []
coch_f_list = []


for pitch_n in range(1,9):
    sr_keyboard, signal_keyboard = wavfile.read('Stimuli_and_acoustics/keyboard_'+str(pitch_n)+'.wav')
    signal_keyboard = signal_keyboard/max(abs(signal_keyboard))
    temp_coch = demo_human_cochleagram_helper(signal_keyboard, sr_keyboard, n, nonlinearity='db')
    coch_k_list.append(temp_coch)
    df_k['k'+str(pitch_n)] = temp_coch.flatten()
    
    sr_string, signal_string = wavfile.read('Stimuli_and_acoustics/string_'+str(pitch_n)+'.wav')
    signal_string = signal_string/max(abs(signal_string))
    temp_coch = demo_human_cochleagram_helper(signal_string, sr_string, n, nonlinearity='db')
    coch_s_list.append(temp_coch)
    df_s['s'+str(pitch_n)] = temp_coch.flatten()
    
    sr_flute, signal_flute = wavfile.read('Stimuli_and_acoustics/flute_'+str(pitch_n)+'.wav')
    signal_flute = signal_flute/max(abs(signal_flute))
    temp_coch =  demo_human_cochleagram_helper(signal_flute, sr_flute, n, nonlinearity='db')
    coch_f_list.append(temp_coch)
    df_f['f'+str(pitch_n)] = temp_coch.flatten()

df_pitch_all = pd.concat([df_k,df_s,df_f],axis=1)

corr_all = df_pitch_all.corr(method='spearman').to_numpy()

corr_mean_pitch = np.empty((8,8))
for n1 in range(3):
    for n2 in range(3):
        corr_mean_pitch += corr_all[0+n1*8:8+n1*8,0+n2*8:8+n2*8]
corr_mean_pitch = corr_mean_pitch/9

# np.save('Stimuli_and_acoustics/cochleagram_db_spearman0723',corr_mean_pitch)


# %% plot cochleagram similarity


fig, axs = plt.subplots(3, 8, figsize=(6, 3.2))
plt.style.use('seaborn-notebook')
# fig.tight_layout()

instru_list = ['Piano', 'Violin', 'Flute']
pitch_list = ['G#6', 'C7', 'E7', 'G#7', 'C8',' E8', 'G#8', 'C9']
coch_all_list = [coch_k_list, coch_s_list, coch_f_list]


sub_list = []

for n_instru in range(3):
    for n_pitch in range(8):
        axs[n_instru,n_pitch].set_title(instru_list[n_instru]+': '+pitch_list[n_pitch], fontsize=5, y=0.95)
        sub_list.append(axs[n_instru,n_pitch].imshow(np.flipud(coch_all_list[n_instru][n_pitch]), aspect='auto', vmin=-60, vmax=0, cmap='magma', origin='lower', interpolation='nearest'))
        
        if (n_instru==2) & (n_pitch==0):
            axs[n_instru,n_pitch].set_yticks([0,260])
            axs[n_instru,n_pitch].set_yticklabels(['50','20k'], fontsize=5)
            axs[n_instru,n_pitch].set_xticks([0,8820,17640])
            axs[n_instru,n_pitch].set_xticklabels([0,0.2,0.4], fontsize=5)
            axs[n_instru,n_pitch].set_ylabel('Hz', fontsize=6)
            axs[n_instru,n_pitch].set_xlabel('time (s)', fontsize=6)
        else:
            axs[n_instru,n_pitch].set_xticks([]) 
            axs[n_instru,n_pitch].set_yticks([])

cb_ax = fig.add_axes([.91,.124,.005,.2])
cbar = fig.colorbar(sub_list[-1],orientation='vertical',cax=cb_ax)
cbar.ax.tick_params(labelsize=5)
cbar.set_label(label='dB', size='small', weight='light', fontsize=6)


# cochleagram similarity
mask = np.triu(np.ones_like(corr_mean_pitch, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(7, 6))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_mean_pitch, mask=mask, cmap='viridis', annot=True, xticklabels=pitch_list, yticklabels=pitch_list,
            square=True, linewidths=.5, cbar_kws={"shrink": .5, 'label': "Spearman's rho"})
ax.set_title('Cochleagram similarity')


# %% plot height and chroma matrices

f, ax = plt.subplots(figsize=(7, 6))
# Draw the heatmap with the mask and correct aspect ratio
array_size = 8
row_nums = np.arange(array_size).reshape(-1, 1)
col_nums = np.arange(array_size)
pitch_dist_mat = np.abs(row_nums - col_nums)/3
sns.heatmap(pitch_dist_mat, mask=mask, cmap='viridis', annot=True, xticklabels=pitch_list, yticklabels=pitch_list,
            square=True, linewidths=.5, cbar_kws={"shrink": .5, 'label': "octave"}, fmt='.2f')
ax.set_title('Pitch height difference')
cbar = ax.collections[0].colorbar
cbar.set_ticks([1,2])



f, ax = plt.subplots(figsize=(7, 6))
# Draw the heatmap with the mask and correct aspect ratio
array_size = 8
row_nums = np.arange(array_size).reshape(-1, 1)
col_nums = np.arange(array_size)
bool_array = np.abs(row_nums - col_nums)%3==0
samePitch_mat = np.where(bool_array, "True", "False")
sns.heatmap(bool_array, mask=mask, cmap='viridis', annot=True, xticklabels=pitch_list, yticklabels=pitch_list, fmt="",
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, cbar=False)
ax.set_title('Same chroma')
# cbar = ax.collections[0].colorbar
# cbar.set_ticks([0, 1])  # Place ticks at the middle of each color bin (0 and 1 in this case)
# cbar.set_ticklabels(['False', 'True'])

