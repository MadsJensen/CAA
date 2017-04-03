# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:17:09 2015

@author: mje
"""
import mne
from mne.minimum_norm import (source_band_induced_power, read_inverse_operator)
import numpy as np
import sys

from my_settings import (tf_folder, subjects_dir, epochs_folder, mne_folder)

subject = sys.argv[1]

# Using the same inverse operator when inspecting single trials Vs. evoked
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr**2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
freqs = np.arange(8, 13, 1)
n_cycle = 4.  # freqs / 3.

conditions = ["ent/left", "ctl/left", "ent/right", "ctl/right"]

# Load data
labels = mne.read_labels_from_annot(
    subject, parc='PALS_B12_Lobes', subjects_dir=subjects_dir)
labels_sel = [labels[9], labels[10]]

inverse_operator = read_inverse_operator(mne_folder + "%s-inv.fif" % subject)
src = mne.read_source_spaces(subjects_dir + "%s/bem/%s-oct-6-src.fif" %
                             (subject, subject))
epochs = mne.read_epochs(epochs_folder + "%s_trial_start-epo.fif" % subject)
# epochs.drop_bad_epochs(reject_params)
# epochs.resample(250, n_jobs=4)

for condition in conditions:
    for label in labels_sel:
        power, itc = source_band_induced_power(
            epochs[condition],
            inverse_operator,
            freqs,
            label,
            baseline=(-0.4, -0.1),
            baseline_mode='ratio',
            n_cycles=n_cycle,
            n_jobs=1)

        power = np.mean(power, axis=0)  # average over sources
        itc = np.mean(itc, axis=0)  # average over sources

        np.save(tf_folder + "%s_%s_%s_%s_%s_source_power.npy" %
                (subject, condition[:3], condition[4:], label.name, method),
                power)
        np.save(tf_folder + "%s_%s_%s_%s_%s_source_itc.npy" %
                (subject, condition[:3], condition[4:], label.name, method),
                itc)
