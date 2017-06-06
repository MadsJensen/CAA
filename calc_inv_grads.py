from my_settings import (mne_folder, epochs_folder)
import sys
import mne
from mne.minimum_norm import make_inverse_operator

subject = sys.argv[1]

fwd = mne.read_forward_solution(mne_folder + "%s-fwd.fif" % subject,
                                surf_ori=False)
fwd = mne.pick_types_forward(fwd, meg="grad", eeg=False)
cov = mne.read_cov(mne_folder + "%s-cov.fif" % subject)
epochs = mne.read_epochs(epochs_folder +
                         "%s_trial_start-epo.fif" % subject,
                         preload=False)


inv = make_inverse_operator(epochs.info, fwd, cov,
                            loose=0.2, depth=0.8)

mne.minimum_norm.write_inverse_operator(mne_folder +
                                        "%s_grad-inv.fif" % subject,
                                        inv)
