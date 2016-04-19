#  -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:45:02 2014.

@author: mje
"""
import mne
import socket
import numpy as np
import os
import sys

from mne.io import Raw

from my_settings import *

subject = "0017"

import matplotlib
matplotlib.use('Agg')

# SETTINGS
n_jobs = 1
l_freq, h_freq = 1, 98  # High and low frequency setting for the band pass
n_freq = 50  # notch filter frequency
decim = 7  # decim value

raw = Raw(save_folder + "%s_filtered_data_mc_raw_tsss.fif" % subject,
          preload=True)
ica = mne.preprocessing.read_ica(save_folder + "%s-ica.fif" % subject)

ica.exclude += [48]

##########################################################################
# Apply the solution to Raw, Epochs or Evoked like this:
raw_ica = ica.apply(raw, copy=False)
ica.save(save_folder + "%s-ica.fif" % subject)  # save ICA componenets
# Save raw with ICA removed
raw_ica.save(save_folder + "%s_filtered_ica_mc_raw_tsss.fif" % subject,
             overwrite=True)
