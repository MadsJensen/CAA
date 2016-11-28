import numpy as np
# import glob
import mne
import pandas as pd

from my_settings import (epochs_folder, tf_folder, subjects_select)

epochs = mne.read_epochs(
    epochs_folder + "0005_target-epo.fif", preload=False)
times = epochs.times

conditions = ["ctl", "ent"]
sides = ["right", "left"]
ROIS = ["rh", "lh"]

from_time = np.abs(times + 0.08).argmin()
to_time = np.abs(times - 0.02).argmin()


data_all = pd.DataFrame()

for subject in subjects_select:
    for condition in conditions:
        for side in sides:
            for roi in ROIS:
                dat = np.load("%s_pow_%s_side_dSPM_Brodmann.17-%s_start.npy" %
                              (subject, condition, side, roi))

                value = dat[:, :, from_time:to_time].mean(axis=0).mean(axis=0)
                
                row = pd.DataFrame([{"subject": subject,
                                     "condition": condition,
                                     "side": side,
                                     "roi": roi,
                                     "power": value}])
                                    
                df = df.append(row, ignore_index=True)

