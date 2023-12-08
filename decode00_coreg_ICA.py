#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:41:36 2021

@author: andrewchang
"""

import mne


import preprocFunc as pf

## run those files one at a time
# runfile('subjPyScript/sub1.py')
# runfile('subjPyScript/sub3.py')
# runfile('subjPyScript/sub4.py')
# runfile('subjPyScript/sub5.py')
# runfile('subjPyScript/sub7.py')
# runfile('subjPyScript/sub8.py')
# runfile('subjPyScript/sub9.py')
# runfile('subjPyScript/sub10.py')
# runfile('subjPyScript/sub12.py')
# runfile('subjPyScript/sub14.py')
# runfile('subjPyScript/sub15.py')
# runfile('subjPyScript/sub16.py')
# runfile('subjPyScript/sub17.py')
# runfile('subjPyScript/sub18.py')


## no subject MRI
### need to coreg!!
# runfile('subjPyScript/sub2.py')
# runfile('subjPyScript/sub6.py')
# runfile('subjPyScript/sub11.py')
# runfile('subjPyScript/sub13.py')

 
    
# %% coregistration
# only need to run it for once
mne.gui.coregistration(subject = subject, subjects_dir=dataMRI_dir, inst = save_dir+subject+'_4coreg.fif')

# if no subject MRI
mne.gui.coregistration(inst = save_dir+subject+'_4coreg.fif')

# %% set up and fit the ICA
# only need to run it for once

# run ICA
epochs_ica, ica = pf.preproICA(raw, save_dir, subject)


# visualize ICA
ica.plot_sources(epochs_ica, show_scrollbars=True)

ica.plot_components()

ica.plot_properties(epochs_ica,picks=list(range(0, 10)))
    