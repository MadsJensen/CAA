# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:55:00 2016

@author: mje
"""

import mne
import numpy as np
from my_settings import (epochs_folder, tf_folder)
from mne.time_frequency import tfr_morlet, write_tfrs
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
        tfrs = tfr_morlet(
            epochs[cond + "/" + side][:10],
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            average=False,
            return_itc=False,
            n_jobs=1)
        tfrs.nave = 1
        write_tfrs(
            tf_folder + "%s_%s_%s-4-tfr.h5" % (subject, cond, side),
            tfrs,
            overwrite=True)
