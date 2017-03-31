#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:51:31 2017

@author: mje
"""

import numpy as np
import mne
import sys
import matplotlib.pyplot as plt
import numpy as np
from mne.stats import permutation_cluster_test

from my_settings import (subjects_select,tf_folder, epochs_folder)


ctl_left = []
ctl_right = []
ent_left = []
ent_right = []

for subject in subjects_select:
    data_right = np.load(tf_folder + "%s_ent_right_itc.npy" % subject)

    epochs = mne.read_epochs(
        epochs_folder + "%s_trial_start-epo.fif" % subject, preload=False)
    selection = mne.read_selection("Left-occipital")
    selection = [f.replace(' ', '') for f in selection]
    left_idx = mne.pick_types(
        epochs.info,
        meg='grad',
        eeg=False,
        eog=False,
        stim=False,
        exclude=[],
        selection=selection)
    
    selection = mne.read_selection("Right-occipital")
    selection = [f.replace(' ', '') for f in selection]
    right_idx = mne.pick_types(
        epochs.info,
        meg='grad',
        eeg=False,
        eog=False,
        stim=False,
        exclude=[],
        selection=selection)
    
    ctl_left.append(data[0, :])
    ctl_right.append(data[1, :])
    ent_left.append(data[2, :])
    ent_right.append(data[3, :])

ctl_left = np.asarray(ctl_left)
ctl_right = np.asarray(ctl_right)
ent_left = np.asarray(ent_left)
ent_right = np.asarray(ent_right)


T_obs, clusters, cluster_pv, H0 = permutation_cluster_test([ctl_left, ctl_right],
                                                           n_permutations=5000)

plt.close('all')
plt.subplot(211)
plt.title("Ctl left v right")
plt.plot(times, ctl_left.mean(axis=0) - ctl_right.mean(axis=0),
         label="ERF Contrast (Event 1 - Event 2)")
plt.ylabel("MEG (T / m)")
plt.legend()
plt.subplot(212)
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_pv[i_c] <= 0.05:
        h = plt.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
    else:
        plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                    alpha=0.3)
hf = plt.plot(times, T_obs, 'g')
plt.legend((h, ), ('cluster p-value < 0.05', ))
plt.xlabel("time (ms)")
plt.ylabel("f-values")
plt.show()