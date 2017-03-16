import sys
import numpy as np
import mne
from mne.decoding import GeneralizationAcrossTime
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from my_settings import (tf_folder, data_path, epochs_folder)

import matplotlib
matplotlib.use('Agg')

subject = sys.argv[1]

# Load epochs from both conditions
data_ctl_left = np.load(tf_folder + "%s_ctl_left-4-tfr.npy" % (subject))
data_ent_left = np.load(tf_folder + "%s_ent_left-4-tfr.npy" % (subject))
data_ctl_right = np.load(tf_folder + "%s_ctl_right-4-tfr.npy" % (subject))
data_ent_right = np.load(tf_folder + "%s_ent_right-4-tfr.npy" % (subject))
epochs = mne.read_epochs(
    epochs_folder + "%s_trial_start-epo.fif" % subject, preload=False)

X = np.vstack([
    np.mean(data_ctl_left, axis=2), np.mean(data_ctl_right, axis=2),
    np.mean(data_ent_left, axis=2), np.mean(data_ent_right, axis=2)
])
y = np.concatenate([
    np.zeros(len(data_ctl_left)), np.ones(len(data_ctl_right)),
    np.ones(len(data_ent_left)) * 2, np.ones(len(data_ent_right)) * 3
])

# Create epochs to use for classification
n_trial, n_chan, n_time = X.shape
events = np.vstack((range(n_trial), np.zeros(n_trial, int), y.astype(int))).T

info = epochs.info
epochs_data = mne.EpochsArray(data=X, info=info, events=events, verbose=False)
epochs_data.times = epochs.times[::4][:-1]

# Crop and downsmample to make it faster
epochs_data.crop(tmin=-0.2, tmax=1)
epochs_data.pick_types(meg="grad")

# Equalise channels and epochs, and concatenate epochs
epochs_data.equalize_event_counts(["0", "1", "2", "3"])

y = epochs_data.events[:, 2]
np.random.shuffle(y)

# Classifier
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1, solver="lbfgs", multi_class="multinomial"))

# Setup the y vector and GAT
gat = GeneralizationAcrossTime(
    predict_mode='mean-prediction', scorer="accuracy", n_jobs=1)

# Fit model
print("fitting GAT")
gat.fit(epochs_data, y=y)

# Scoring
print("Scoring GAT")
gat.score(epochs_data, y=y)

# Save model
joblib.dump(gat, data_path + "decode_time_gen/%s_gat_all_grad.jl" % subject)

# make matrix plot and save it
fig = gat.plot(cmap="viridis", title="Temporal Gen for subject: %s" % subject)
fig.savefig(data_path + "decode_time_gen/%s_gat_matrix_all_grad.png" % subject)

fig = gat.plot_diagonal(
    chance=0.5, title="Temporal Gen for subject: %s" % subject)
fig.savefig(data_path + "decode_time_gen/%s_gat_diagonal_all_grad.png" %
            subject)
