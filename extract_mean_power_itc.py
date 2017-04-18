# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:10:13 2017

@author: mje
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import mne

from my_settings import (subjects_select, tf_folder, epochs_folder)

plt.style.use("seaborn")

epochs = mne.read_epochs(
    epochs_folder + "0005_trial_start-epo.fif", preload=False)
times = (epochs.times[::4][:-1])

from_time = np.abs(times + 0).argmin()
to_time = np.abs(times - 0.7).argmin()

mean_power_df = pd.DataFrame()

for subject in subjects_select:
    data = np.load(tf_folder + "%s_ali_pow_source_Lobes.npy" % subject)
    ctl_left = data[0, from_time:to_time].mean()
    ctl_right = data[1, from_time:to_time].mean()
    ent_left = data[2, from_time:to_time].mean()
    ent_right = data[3, from_time:to_time].mean()
    
    tmp_df = pd.DataFrame([{"subject": subject,
                            "ctl_left": ctl_left,
                            "ctl_right": ctl_right,
                            "ent_left": ent_left,
                            "ent_right": ent_right}])
    
    mean_power_df = mean_power_df.append(tmp_df, ignore_index=True)

mean_ali_long = pd.melt(mean_power_df, id_vars=["subject"])
mean_ali_long.columns  = ['subject', 'variable', 'ALI']

mean_ali_long.to_csv(tf_folder + "mean_ali_long.csv", index=False)

# ITC data
mean_itc_df = pd.DataFrame()

for subject in subjects_select:
    data = np.load(tf_folder + "%s_ali_itc_source_Lobes.npy" % subject)
    ctl_left = data[0, from_time:to_time].mean()
    ctl_right = data[1, from_time:to_time].mean()
    ent_left = data[2, from_time:to_time].mean()
    ent_right = data[3, from_time:to_time].mean()
    
    tmp_df = pd.DataFrame([{"subject": subject,
                            "ctl_left": ctl_left,
                            "ctl_right": ctl_right,
                            "ent_left": ent_left,
                            "ent_right": ent_right}])
    
    mean_itc_df = mean_itc_df.append(tmp_df, ignore_index=True)    

mean_itc_long = pd.melt(mean_itc_df, id_vars=["subject"])
mean_itc_long.columns  = ['subject', 'variable', 'ITC']

mean_itc_long.to_csv(tf_folder + "mean_itc_long.csv", index=False)
