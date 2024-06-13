#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:33:00 2024

@author: andrewchang
"""

import os
import numpy as np
import mne
import preprocFunc as pf
import matplotlib.pyplot as plt


epochs_reconst = []
for file_num in [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]:
    filename = 'subjPyScript/sub'+str(file_num)+'.py'
    with open(filename, 'r') as file:
        code = file.read()
    
    # Execute the code from the .py file
    exec(code)

    epochs_reconst.append(pf.get_ERP(raw, ica_exclude, save_dir, subject).average())
    

grand_average = mne.grand_average(epochs_reconst)
grand_average.save('save_fif/ERP_grant_average_0613')
    
fig = grand_average.crop(tmin=-0.1).plot_joint(times=[0.1,0.2,0.3,0.4], title='') 
plt.savefig('ERP.png', format='png', dpi=600)