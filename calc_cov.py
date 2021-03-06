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

# fig = epochs.plot_drop_log(subject=subject, show=False)
# fig.savefig(epochs_folder + "pics/%s_drop_log.png" % subject)

# Make noise cov
cov = compute_covariance(epochs, tmin=None, tmax=0, method="shrunk")
mne.write_cov(mne_folder + "%s-cov.fif" % subject, cov)
