import numpy as np
# import glob
import mne
import pandas as pd

from my_settings import (epochs_folder, tf_folder)

subjects_select = ["0005", "0006", "0007", "0008", "0009", "0010",
                   "0011", "0015", "0016", "0017", "0020", "0021",
                   "0022", "0024", "0025"]

epochs = mne.read_epochs(
    epochs_folder + "0005_target-epo.fif", preload=False)
times = epochs.times

conditions = ["ctl", "ent"]
sides = ["right", "left"]
ROIS = ["rh", "lh"]

from_time = np.abs(times + 0.08).argmin()
to_time = np.abs(times - 0.02).argmin()


df = pd.DataFrame()

for subject in subjects_select:
    for condition in conditions:
        for side in sides:
            for roi in ROIS:
                dat = np.load(tf_folder + "%s_itc_%s_%s_dSPM_Brodmann.17-%s_target.npy" %
                              (subject, condition, side, roi))

                value = dat[:, :, from_time:to_time].mean(axis=0).mean(axis=0).mean(axis=0)
                
                row = pd.DataFrame([{"subject": subject,
                                     "condition": condition,
                                     "side": side,
                                     "roi": roi,
                                     "itc": value}])
                                    
                df = df.append(row, ignore_index=True)

