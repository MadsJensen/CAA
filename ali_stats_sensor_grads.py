#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:14:24 2017

@author: mje
"""
import matplotlib.pyplot as plt
import numpy as np
from mne.stats import permutation_cluster_test
import mne 

from my_settings import (subjects_select,tf_folder, epochs_folder)


ctl_left = []
ctl_right = []
ent_left = []
ent_right = []

for subject in subjects_select:
    data = np.load(tf_folder + "%s_ali.npy" % subject)
    ctl_left.append(data[0, :])
    ctl_right.append(data[1, :])
    ent_left.append(data[2, :])
    ent_right.append(data[3, :])

ctl_left = np.asarray(ctl_left)
ctl_right = np.asarray(ctl_right)
ent_left = np.asarray(ent_left)
ent_right = np.asarray(ent_right)

epochs = mne.read_epochs(
    epochs_folder + "0005_trial_start-epo.fif", preload=False)
times = (epochs.times[::4][:-1])*1e3

T_obs, clusters, cluster_pv, H0 = permutation_cluster_test([ctl_left, ctl_right],
                                                           n_permutations=5000)

plt.figure()
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


d_ctl_ali = ctl_left - ctl_right
d_ent_ali = ent_left - ent_right

T_obs, clusters, cluster_pv, H0 = permutation_cluster_test([d_ctl_ali, d_ent_ali],
                                                           n_permutations=5000)

plt.figure()
plt.subplot(211)
plt.title("Ctl v ent ")
plt.plot(times, d_ctl_ali.mean(axis=0) - d_ent_ali.mean(axis=0),
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


T_obs, clusters, cluster_pv, H0 = permutation_cluster_test([ent_left, ent_right],
                                                           n_permutations=5000)

plt.figure()
plt.subplot(211)
plt.title("Ent left v right")
plt.plot(times, d_ctl_ali.mean(axis=0) - d_ent_ali.mean(axis=0),
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