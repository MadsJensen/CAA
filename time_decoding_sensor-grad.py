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

epochs_clt_left = epochs["ctl/left"].copy()
epochs_clt_right = epochs["ctl/right"].copy()
epochs_ent_left = epochs["ent/left"].copy()
epochs_ent_right = epochs["ent/right"].copy()

del epochs

epochs_clt_left.events[:2] = 0
epochs_clt_right.events[:2] = 1
epochs_ent_left.events[:2] = 2
epochs_ent_right.events[:2] = 3

epochs_clt_left.event_id = {"0": 0}
epochs_clt_right.event_id = {"1": 1}
epochs_ent_left.event_id = {"2": 2}
epochs_ent_right.event_id = {"3": 3}

epochs_data = mne.concatenate_epochs(
    [epochs_clt_left, epochs_clt_right, epochs_ent_left, epochs_ent_right])

# Equalise channels and epochs, and concatenate epochs
epochs_data.equalize_event_counts(["0", "1", "2", "3"])

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
