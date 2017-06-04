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

conditions = ["ctl", "ent"]
sides = ["lh", "rh"]
targets = ["left", "right"]

for subject in subjects_select:
    for cond in conditions:
        for side in sides:
            for target in targets:
                data = np.load(
                    tf_folder +
                    "%s_%s_%s_LOBE.OCCIPITAL-%s_MNE_source_power_snr_3.npy" %
                    (subject, cond, target, side))

                data = np.percentile(data, 90, axis=0)
                data = data[from_time:to_time].mean()

                tmp_df = pd.DataFrame([{
                    "subject": subject,
                    "side": side,
                    "target": target,
                    "condition": cond,
                    "pow": data
                }])

                mean_power_df = mean_power_df.append(tmp_df, ignore_index=True)

mean_power_df.to_csv(tf_folder + "mean_pow_long.csv", index=False)

# ITC data
mean_itc_df = pd.DataFrame()

for subject in subjects_select:
    for cond in conditions:
        for side in sides:
            for target in targets:
                data = np.load(
                    tf_folder +
                    "%s_%s_%s_LOBE.OCCIPITAL-%s_MNE_source_itc_snr_3.npy" %
                    (subject, cond, target, side))

                data = np.percentile(data, 90, axis=0)
                data = data[from_time:to_time].mean()

                tmp_df = pd.DataFrame([{
                    "subject": subject,
                    "side": side,
                    "target": target,
                    "condition": cond,
                    "pow": data
                }])

                mean_itc_df = mean_itc_df.append(tmp_df, ignore_index=True)

mean_itc_df.to_csv(tf_folder + "mean_itc_long.csv", index=False)
