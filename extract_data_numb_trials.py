# -*- coding: utf-8 -*-
"""
@author: mje
"""

from my_settings import
import mne
import numpy as np
import pandas as pd

sides = ["left", "right"]
conditions = ["ctl", "ent"]
rois = ["lh", "rh"]
corr = ["correct", "incorrect"]
phase = ["in_phase", "out_phase"]

columns_keys = ["subject", "condition_type", "condition_side", "ROI",
                "correct", "mean"]
df = pd.DataFrame(columns=columns_keys)

for subject in subjects_select:
    epochs = mne.read_epochs(epochs_folder + "%s_target-epo.fif" % subjject,
                             preload=False)

    for condition in conditions:
        for side in sides:
            for roi in rois:
                for cor in corr:
                    for p in phase:
                        row = pd.DataFrame([{"subject": subject,
                                             "condition_type": condition,
                                             "condition_side": side,
                                             "ROI": roi,
                                             "correct": cor,
                                             "phase": p,
                                             "n": len(epochs[condition + "/" +
                                                             side + "/" +
                                                             roi + "/" +
                                                             cor + "/" + p])}])
                        df = df.append(row, ignore_index=True)

df.to_csv(data_path + "alpha_mean_n_data_extracted_phase_target.csv",
          index=False)
