# -*- coding: utf-8 -*-
"""
.. _stats_cluster_sensors_2samp_spatial:

=====================================================
Spatiotemporal permutation F-test on full sensor data
=====================================================

Tests for differential evoked responses in at least
one condition using a permutation clustering test.
The FieldTrip neighbor templates will be used to determine
the adjacency between sensors. This serves as a spatial prior
to the clustering. Significant spatiotemporal clusters will then
be visualized using custom matplotlib code.
"""
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne.viz import plot_topomap
import cPickle as pickle

import mne
from mne.stats import spatio_temporal_cluster_test
from mne.channels import read_ch_connectivity

from my_settings import *

###############################################################################
X_ctl_left = np.empty([len(subjects_select[:-1]), 501, 204])
X_ctl_right = np.empty([len(subjects_select[:-1]), 501, 204])
X_ent_left = np.empty([len(subjects_select[:-1]), 501, 204])
X_ent_right = np.empty([len(subjects_select[:-1]), 501, 204])

for j, subject in enumerate(subjects_select[:-1]):
    epochs = mne.read_epochs(epochs_folder +
                             '%s_hilbert_pow_trial_start-epo.fif' % subject)

    epochs.pick_types(meg="grad")

    X_ctl_left[j, :, :] = epochs["ctl/left"].average().data.T
    X_ctl_right[j, :, :] = epochs["ctl/right"].average().data.T
    X_ent_left[j, :, :] = epochs["ent/left"].average().data.T
    X_ent_right[j, :, :] = epochs["ent/right"].average().data.T


X = [X_ctl_left, X_ctl_right, X_ent_left, X_ent_right]

###############################################################################
# load FieldTrip neighbor definition to setup sensor connectivity
connectivity, ch_names = read_ch_connectivity('neuromag306planar')

# set cluster threshold
# set family-wise p-value
p_accept = 0.05

cluster_stats = spatio_temporal_cluster_test(X, n_permutations=5000,
                                             tail=0,
                                             n_jobs=4,
                                             connectivity=connectivity)

T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]

pickle.dump(cluster_stats, open(result_dir +
                                "/cluster_stats_pow_grad.p", "wb"))
