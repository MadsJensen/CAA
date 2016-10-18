import mne
import sys

from mne import compute_covariance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from my_settings import *

reject = dict(
    grad=4000e-13,  # T / m (gradiometers)
    mag=4e-12,  # T (magnetometers)
    eeg=180e-6  #
)

subject = sys.argv[1]

epochs = mne.read_epochs(epochs_folder + "%s_trial_start-epo.fif" % subject)
epochs.drop_bad_epochs(reject_params)

# fig = epochs.plot_drop_log(subject=subject, show=False)
# fig.savefig(epochs_folder + "pics/%s_drop_log.png" % subject)

# Make noise cov
cov = mne.compute_covariance(
    epochs,
    method=['empirical', 'shrunk'],
    tmin=-0.8,
    tmax=0.0,
    return_estimators=True,
    verbose=True)

evoked = epochs.average()
fig = evoked.plot_white(cov, show=False)
fig.subtitle("subject: %s" % subject)
fig.savefig(mne_folder + "sub_%.png" % subject)
