import mne
import sys

from mne import compute_covariance

from my_settings import (epochs_folder, mne_folder)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reject = dict(
    grad=4000e-13,  # T / m (gradiometers)
    mag=4e-12,  # T (magnetometers)
    eeg=180e-6  #
)

subject = sys.argv[1]

epochs = mne.read_epochs(epochs_folder + "%s_trial_start-epo.fif" % subject)
epochs.drop_bad(reject)

# Make noise cov
cov = mne.compute_covariance(
    epochs,
    method=['empirical', 'shrunk'],
    tmin=-0.5,
    tmax=0.0,
    return_estimators=True,
    verbose=True)

evoked = epochs.average()
fig = evoked.plot_white(cov, show=False)
fig.suptitle("subject: %s" % subject)
fig.savefig(mne_folder + "cov_plots/sub_%s.png" % subject)
