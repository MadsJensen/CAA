# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:55:00 2016

@author: mje
"""

import mne
import numpy as np
from my_settings import (epochs_folder, tf_folder)
from mne.time_frequency import tfr_array_morlet
import sys

subject = sys.argv[1]

freqs = np.arange(8, 13, 1)  # define frequencies of interest
n_cycles = 4.  # freqs / 2.  # different number of cycle per frequency

sides = ["left", "right"]
conditions = ["ctl", "ent"]

epochs = mne.read_epochs(
    epochs_folder + "%s_trial_start-epo.fif" % subject, preload=True)
epochs.resample(250)

for cond in conditions:
    for side in sides:
        power = tfr_array_morlet(
            epochs[cond + "/" + side],
            sfreq=epochs.info["sfreq"],
            frequencies=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            output="complex",
            n_jobs=1)
        np.save(tf_folder + "%s_%s_%s-4-complex-tfr.npy" %
                (subject, cond, side), power)
