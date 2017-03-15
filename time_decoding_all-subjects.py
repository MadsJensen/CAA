import numpy as np
import mne
from mne import create_info
from mne.decoding import GeneralizationAcrossTime

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from my_settings import (tf_folder, data_path, subjects_select)

import matplotlib
matplotlib.use('Agg')

ave_ctl_left = []
ave_ctl_right = []
ave_ent_left = []
ave_ent_right = []

for subject in subjects_select:
    print("loading subject: %s" % subject)
    epochs = mne.read_epochs(tf_folder + "%s_power-epo.fif" % subject)
    # Crop and downsmample to make it faster
    epochs.crop(tmin=-0.2, tmax=1)
    epochs.pick_types(meg="grad")

    # Equalise channels and epochs, and concatenate epochs
    epochs.equalize_event_counts(
        ["ctl/left", "ctl/right", "ent/left", "ent/right"])

    ave_ctl_left.append(epochs["ctl/left"].average())
    ave_ctl_right.append(epochs["ctl/right"].average())
    ave_ent_left.append(epochs["ent/left"].average())
    ave_ent_right.append(epochs["ent/right"].average())

ave_ctl_left = np.asarray(ave_ctl_left)
ave_ctl_right = np.asarray(ave_ctl_right)
ave_ent_left = np.asarray(ave_ent_left)
ave_ent_right = np.asarray(ave_ent_right)

print("making X & y")
X = np.vstack([ave_ctl_left, ave_ctl_right, ave_ent_left, ave_ent_right])
y = np.concatenate([
    np.zeros(len(ave_ctl_left)), np.ones(len(ave_ctl_right)),
    np.ones(len(ave_ent_left)) * 2, np.ones(len(ave_ent_right)) * 3
])

n_trial, n_chan, n_time = X.shape
events = np.vstack((range(n_trial), np.zeros(n_trial, int), y.astype(int))).T

chan_names = ['MEG %i' % chan for chan in range(n_chan)]
chan_types = ['mag'] * n_chan
sfreq = 250
info = create_info(chan_names, sfreq, chan_types)

print("Creating epochs")
epochs_data = mne.EpochsArray(data=X, info=info, events=events, verbose=False)
epochs_data.times = epochs.times[::4][:-1]

# Classifier
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1, solver="lbfgs", multi_class="multinomial"))

# Setup the y vector and GAT
gat = GeneralizationAcrossTime(
    predict_mode='mean-prediction', scorer="accuracy", n_jobs=1)

# Fit model
print("fitting GAT")
gat.fit(epochs_data)

# Scoring
print("Scoring GAT")
gat.score(epochs_data)

# Save model
print("Saving model")
joblib.dump(gat,
            data_path + "decode_time_gen/%s_gat_all-subs_grad.jl" % subject)

# make matrix plot and save it
fig = gat.plot(cmap="viridis", title="Temporal Gen for subject: %s" % subject)
fig.savefig(data_path + "decode_time_gen/%s_gat_matrix_all-subs_grad.png" %
            subject)

fig = gat.plot_diagonal(
    chance=0.5, title="Temporal Gen for subject: %s" % subject)
fig.savefig(data_path + "decode_time_gen/%s_gat_diagonal_all-subs_grad.png" %
            subject)
