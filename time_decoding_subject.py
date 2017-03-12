import sys
import numpy as np
import mne
from mne.decoding import GeneralizationAcrossTime
from sklearn.externals import joblib

from my_settings import (tf_folder, data_path, epochs_folder)

import matplotlib
matplotlib.use('Agg')

subject = sys.argv[1]

# Load epochs from both conditions
data_ctl_left = np.load(tf_folder + "%s_ctl_left-4-tfr.npy" % (subject))
data_ctl_right = np.load(tf_folder + "%s_ctl_right-4-tfr.npy" % (subject))
epochs = mne.read_epochs(
    epochs_folder + "%s_trial_start-epo.fif" % subject, preload=True)
epochs.resample(250)

X = np.vstack([
    np.mean(np.abs(data_ctl_left)**2, axis=2),
    np.mean(np.abs(data_ctl_right)**2, axis=2)
])
y = np.concatenate(
    [np.zeros(len(data_ctl_left)), np.ones(len(data_ctl_right))])

# Create epochs to use for classification
n_trial, n_chan, n_time = X.shape
events = np.vstack((range(n_trial), np.zeros(n_trial, int), y.astype(int))).T
sfreq = 250

info = epochs.info
epochs_data = mne.EpochsArray(data=X, info=info, events=events, verbose=False)
epochs_data.times = epochs.times

# Equalise channels and epochs, and concatenate epochs
mne.epochs.equalize_epoch_counts(epochs_data)

# Crop and downsmample to make it faster
epochs.crop(tmin=None, tmax=1)

# Setup the y vector and GAT
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
