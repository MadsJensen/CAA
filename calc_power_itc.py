

from my_settings import *
import mne
import sys
import numpy as np
import matplotlib.pyplot as plt

from mne.minimum_norm import read_inverse_operator, source_induced_power

subject = sys.argv[1]

epochs = mne.read_epochs(epochs_folder + "%s_trial_start-epo.fif" 
                         % subject)
epochs.drop_bad_epochs(reject_params)

inv = mne.minimum_norm.read_inverse_operator(mne_folder + "%s-inv.fif"
                                             % subject)

labels = mne.read_labels_from_annot(subject, parc='PALS_B12_Brodmann',
                                    regexp="Bro",
                                    subjects_dir=subjects_dir)
labels_selc = labels[6], labels[7]

frequencies = np.arange(8, 13, 1)  # define frequencies of interest
n_cycles = frequencies / 3.  # different number of cycle per frequency

sides = ["left", "right"]
conditions = ["ctl", "ent"]

for cond in conditions:
    for j, side in enumerate(sides):
        power, itc = source_induced_power(epochs[cond + "/" + side],
                                          inv,
                                          frequencies,
                                          labels_selc[j],
                                          baseline=(-0.3, 0),
                                          baseline_mode='zscore',
                                          n_cycles=n_cycles,
                                          n_jobs=1)
        np.save(tf_folder + "%s_pow_%s_%s_%s.npy" % (subject, cond, side,
                                                     labels_selc[j].name),
                power)
        np.save(tf_folder + "%s_itc_%s_%s_%s_%s.npy" % (subject, cond, side,
                                                        labels_selc[j].name),
                itc)
