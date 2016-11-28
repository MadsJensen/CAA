# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:17:09 2015

@author: mje
"""
import mne
from mne.minimum_norm import (apply_inverse_epochs, read_inverse_operator)
from mne.time_frequency import morlet
import numpy as np
import sys

from my_settings import *

subject = sys.argv[1]

# Using the same inverse operator when inspecting single trials Vs. evoked
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr**2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
freqs = np.arange(8, 13, 1)
n_cycle = freqs / 3.

conditions = ["ent/left", "ctl/left", "ent/right", "ctl/right"]

# Load data
labels = mne.read_labels_from_annot(
    subject, parc='PALS_B12_Brodmann', regexp="Bro", subjects_dir=subjects_dir)
labels_sel = [labels[6], labels[7]]

inverse_operator = read_inverse_operator(mne_folder + "%s-inv.fif" % subject)
src = mne.read_source_spaces(subjects_dir + "%s/bem/%s-oct-6-src.fif" % (
    subject, subject))
epochs = mne.read_epochs(epochs_folder + "%s_trial_start-epo.fif" % subject)
# epochs.drop_bad_epochs(reject_params)
# epochs.resample(250, n_jobs=4)

for condition in conditions[:1]:
    stcs = apply_inverse_epochs(
        epochs[condition],
        inverse_operator,
        lambda2,
        method,
        pick_ori="normal")

    for label in labels_sel:
        label_ts = []
        for j in range(len(stcs[:1])):
            ts = mne.extract_label_time_course(
                stcs[j], labels=label, src=src, mode="pca_flip")
            ts = np.squeeze(ts)
            ts *= np.sign(ts[np.argmax(np.abs(ts))])
            label_ts.append(ts)

        label_ts = np.asarray(label_ts)
        tfr = morlet(
            label_ts,
            epochs.info["sfreq"],
            freqs,
            # use_fft=True,
            n_cycles=n_cycle)

        np.save(tf_folder + "%s_%s_%s_%s_%s-tfr" %
                (subject, condition[:3], condition[4:], label.name, method),
                tfr)
        np.save(tf_folder + "%s_%s_%s_%s_%s-ts" %
                (subject, condition[:3], condition[4:], label.name, method),
                label_ts)

    del stcs
    del tfr
