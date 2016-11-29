from my_settings import (epochs_folder, tf_folder, subjects_dir, mne_folder)
import mne
import sys
import numpy as np
import pandas as pd

from mne.minimum_norm import read_inverse_operator, source_induced_power

subject = sys.argv[1]

epochs = mne.read_epochs(epochs_folder + "%s_target-epo.fif" % subject)

inv = read_inverse_operator(mne_folder + "%s-inv.fif" % subject)

labels = mne.read_labels_from_annot(
    subject, parc='PALS_B12_Lobes',
    # regexp="Bro",
    subjects_dir=subjects_dir)
labels_selc = labels[9], labels[10]

frequencies = np.arange(8, 13, 1)  # define frequencies of interest
n_cycles = frequencies / 3.  # different number of cycle per frequency
method = "MNE"

sides = ["left", "right"]
conditions = ["ctl", "ent"]
cor = ["correct", "incorrect"]
phase = ["in_phase", "out_phase"]
congrunet = ["cong", "incong"]

columns_keys = ["subject", "side", "condition", "phase", "ROI"]
df = pd.DataFrame(columns=columns_keys)

for label in labels_selc:
    for cond in conditions:
        for j, side in enumerate(sides):
            power, itc = source_induced_power(
                epochs[cond + "/" + side],
                inv,
                frequencies,
                label=label,
                method=method,
                pick_ori=None,
                use_fft=True,
                baseline=(-1.4, -1.1),
                baseline_mode='zscore',
                n_cycles=n_cycles,
                pca=True,
                n_jobs=1)
            np.save(tf_folder + "%s_pow_%s_%s_%s_%s_target.npy" %
                    (subject, cond, side, method, label.name), power)
            np.save(tf_folder + "%s_itc_%s_%s_%s_%s_target.npy" %
                    (subject, cond, side, method, label.name), itc)

            n = len(epochs[cond + "/" + side])

            row = pd.DataFrame([{
                "subject": subject,
                "side": side,
                "condition": cond,
                "ROI": label.name,
            }])
            df = df.append(row, ignore_index=True)

df.to_csv(tf_folder + "%s_tf_target_epochs.csv" % subject)
