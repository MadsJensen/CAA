import numpy as np
# import glob
import mne
import pandas as pd
import matplotlib.pyplot as plt

from my_settings import (epochs_folder, tf_folder)

subjects_select = [
    "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0015", "0016",
    "0017", "0020", "0021", "0022", "0024", "0025"
]

epochs = mne.read_epochs(epochs_folder + "0005_target-epo.fif", preload=False)
times = epochs.times

conditions = ["ctl", "ent"]
sides = ["right", "left"]
ROIS = ["rh", "lh"]

from_time = np.abs(times + 0.08).argmin()
to_time = np.abs(times - 0.02).argmin()

df = pd.DataFrame()

for condition in conditions:
    for side in sides:
        for roi in ROIS:
            plt.figure()
            for subject in subjects_select:
                dat = np.load(tf_folder +
                              "%s_pow_%s_%s_dSPM_Brodmann.17-%s_target.npy" % (
                                  subject, condition, side, roi))
                plt.plot(times, dat.mean(axis=0).mean(axis=0))
                plt.title("%s %s %s" % (condition, side, roi))
