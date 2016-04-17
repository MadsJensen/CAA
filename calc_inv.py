from my_settings import *
import sys
import mne
from mne.minimum_norm import make_inverse_operator

subject = sys.argv[1]

fwd = mne.read_forward_solution(mne_folder + "%s-fwd.fif" % subject)
cov = mne.read_cov(mne_folder + "%s-cov.fif" % subject)
epochs = mne.read_epochs(epochs_folder +
                         "%s_filtered_ica_mc_tsss-epo.fif" % subject,
                         preload=False)
epochs.drop_bad_epochs(reject_params)

inv = make_inverse_operator(epochs.info, fwd, cov,
                            loose=0.2, depth=0.8)

mne.minimum_norm.write_inverse_operator(mne_folder +
                                        "%s-inv.fif" % subject,
                                        inv)