import sys
import numpy as np
import mne
from mne.decoding import GeneralizationAcrossTime
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from my_settings import (data_path, epochs_folder)

import matplotlib
matplotlib.use('Agg')

subject = sys.argv[1]

# Load epochs from both conditions
epochs = mne.read_epochs(
    epochs_folder + "%s_trial_start-epo.fif" % subject, preload=True)

# Crop and downsmample to make it faster
epochs.crop(tmin=-0.2, tmax=1)
epochs.pick_types(meg="grad")

# Equalise channels and epochs, and concatenate epochs
epochs.equalize_event_counts(
    ["clt/left", "ctl/right", "ent/left", "ent/right"])

# Classifier
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1, solver="lbfgs", multi_class="multinomial"))

# Setup the y vector and GAT
gat = GeneralizationAcrossTime(
    predict_mode='mean-prediction', scorer="accuracy", n_jobs=1)

# Fit model
print("fitting GAT")
gat.fit(epochs)

# Scoring
print("Scoring GAT")
gat.score(epochs)

# Save model
joblib.dump(gat,
            data_path + "decode_time_gen/%s_gat_allsensor-grad.jl" % subject)

# make matrix plot and save it
fig = gat.plot(cmap="viridis", title="Temporal Gen for subject: %s" % subject)
fig.savefig(data_path + "decode_time_gen/%s_gat_matrix_allsensor-grad.png" %
            subject)

fig = gat.plot_diagonal(
    chance=0.5, title="Temporal Gen for subject: %s" % subject)
fig.savefig(data_path + "decode_time_gen/%s_gat_diagonal_allsensor-grad.png" %
            subject)
