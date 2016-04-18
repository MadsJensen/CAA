import mne
import sys

from mne import compute_covariance

from my_settings import *

subject = sys.argv[1]

epochs = mne.read_epochs(epochs_folder + "%s_trial_start-epo.fif" % subject)

# Make noise cov
cov = compute_covariance(epochs, tmin=None, tmax=-0.2,
                         method="shrunk")
mne.write_cov(mne_folder + "%s-cov.fif" % subject, cov)
