import sys
import numpy as np
import mne
from mne.decoding import GeneralizationAcrossTime
from mne.time_frequency import read_tfrs
from sklearn.externals import joblib

from my_settings import (tf_folder, data_path, epochs_folder)

import matplotlib
matplotlib.use('Agg')

subject = sys.argv[1]

# Load epochs from both conditions
tfrs_ctl_left = read_tfrs(tf_folder + "%s_ctl_left-4-tfr.hd5" % (subject))
tfrs_ctl_right = read_tfrs(tf_folder + "%s_ctl_right-4-tfr.hd5" % (subject))
epochs = mne.read_epochs(
    epochs_folder + "%s_trial_start-epo.fif" % subject, preload=False)

# Fix the events for the plan epochs so they can be concatenated
epochs_ctl_left = mne.EpochsArray(tfrs_ctl_left.data.mean(axis=2), epochs.info)
epochs_ctl_right = mne.EpochsArray(
    tfrs_ctl_right.data.mean(axis=2), epochs.info)

# Equalise channels and epochs, and concatenate epochs
mne.equalize_channels([epochs_ctl_left, epochs_ctl_right])
mne.epochs.equalize_epoch_counts([epochs_ctl_left, epochs_ctl_right])

# Dirty hack # TODO: Check this from the Maxfilter side
# epochs_classic.info['dev_head_t'] = epochs_plan.info['dev_head_t']

epochs_tfr = mne.concatenate_epochs([epochs_ctl_left, epochs_ctl_right])

# Crop and downsmample to make it faster
epochs.crop(tmin=None, tmax=1)

# Setup the y vector and GAT
y = np.concatenate(
    (np.zeros(len(epochs["press"])), np.ones(len(epochs["plan"]))))
gat = GeneralizationAcrossTime(
    predict_mode='mean-prediction', scorer="roc_auc", n_jobs=1)

# Fit model

# Scoring and visualise result
gat.score(epochs, y=y)

# Save model
joblib.dump(gat, data_path + "decode_time_gen/%s_gat_2.jl" % subject)

fig = gat.plot(
    title="Temporal Gen (Classic vs planning): left to right sub: %s" %
    subject)
fig.savefig(data_path + "decode_time_gen/%s_gat_matrix_2.png" % subject)
