# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:58:36 2016

@author: mje
"""

from my_settings import *
import mne
import numpy as np
import pandas as pd

epochs = mne.read_epochs(epochs_folder + "0005_trial_start-epo.fif",
                         preload=False)
times = epochs.times


from_time = np.abs(times + 0).argmin()
to_time = np.abs(times - 0.7).argmin()

sides = ["left", "right"]
conditions = ["ctl", "ent"]
rois = ["lh", "rh"]

columns_keys = ["subject", "condition_type", "condition_side", "ROI",
                "mean"]
df = pd.DataFrame(columns=columns_keys)

for subject in subjects_select:
    for condition in conditions:
        for side in sides:
            for roi in rois:
                data = np.load(tf_folder +
                    "%s_pow_%s_%s_dSPM_Brodmann.17-%s_n4_bs.npy" % (subject,
                                                                    condition,
                                                                    side,
                                                                    roi))
                data = data[:, :, from_time:to_time].mean(axis=1).mean(axis=1)
                idx = np.arange(len(data))
                for i in idx:
                    row = pd.DataFrame([{"subject": subject,
                                         "condition_type": condition,
                                         "condition_side": side,
                                         "ROI": roi,
                                         "mean": data[i]}])
                    df = df.append(row, ignore_index=True)


df.to_csv(data_path + "alpha_mean_pow_data_extracted.csv", index=False)


columns_keys = ["subject", "condition_type", "condition_side", "ROI",
                "mean"]
df = pd.DataFrame(columns=columns_keys)

for subject in subjects_select:
    for condition in conditions:
        for side in sides:
            for roi in rois:
                data = np.load(tf_folder +
                    "%s_itc_%s_%s_dSPM_Brodmann.17-%s_n4_bs.npy" % (subject,
                                                                    condition,
                                                                    side,
                                                                    roi))
                data = data[:, :, from_time:to_time].mean(axis=1).mean(axis=1)
                idx = np.arange(len(data))
                for i in idx:
                    row = pd.DataFrame([{"subject": subject,
                                         "condition_type": condition,
                                         "condition_side": side,
                                         "ROI": roi,
                                         "mean": data[i]}])
                    df = df.append(row, ignore_index=True)


df.to_csv(data_path + "alpha_mean_itc_data_extracted.csv", index=False)
