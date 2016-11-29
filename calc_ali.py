from my_settings import (tf_folder, epochs_folder, subjects_select, result_dir)
import mne
import numpy as np
import pandas as pd

epochs = mne.read_epochs(epochs_folder + "0005_target-epo.fif", preload=False)
times = epochs.times

from_time = np.abs(times + 0.1).argmin()
to_time = np.abs(times - 0.1).argmin()

sides = ["left", "right"]
conditions = ["ctl", "ent"]
rois = ["lh", "rh"]
corr = ["correct", "incorrect"]
phase = ["in_phase", "out_phase"]

columns_keys = ["subject", "type", "side", "correct", "phase", "ALI_pow"]

df = pd.DataFrame(columns=columns_keys)

for subject in subjects_select:
    print("Working on subject: %s" % subject)
    for cor in corr:
        for p in phase:
            ctl_lc_lr = np.load(tf_folder + "%s_pow_ctl_left_dSPM_%s_%s" % (
                subject, cor, p) + "_Brodmann.17-lh_target.npy")
            ctl_lc_rr = np.load(tf_folder + "%s_pow_ctl_left_dSPM_%s_%s" % (
                subject, cor, p) + "_Brodmann.17-rh_target.npy")
            ctl_lc_lr = ctl_lc_lr.mean(axis=0).mean(axis=0)
            ctl_lc_rr = ctl_lc_rr.mean(axis=0).mean(axis=0)

            ali_ctl_left = ((ctl_lc_lr - ctl_lc_rr) / (ctl_lc_lr + ctl_lc_rr))

            ctl_rc_lr = np.load(tf_folder + "%s_pow_ctl_right_dSPM_%s_%s" % (
                subject, cor, p) + "_Brodmann.17-lh_target.npy")
            ctl_rc_rr = np.load(tf_folder + "%s_pow_ctl_right_dSPM_%s_%s" % (
                subject, cor, p) + "_Brodmann.17-rh_target.npy")
            ctl_rc_lr = ctl_rc_lr.mean(axis=0).mean(axis=0)
            ctl_rc_rr = ctl_rc_rr.mean(axis=0).mean(axis=0)

            ali_ctl_right = ((ctl_rc_lr - ctl_rc_rr) / (ctl_rc_lr + ctl_rc_rr))

            ent_lc_lr = np.load(tf_folder + "%s_pow_ent_left_dSPM_%s_%s" % (
                subject, cor, p) + "_Brodmann.17-lh_target.npy")
            ent_lc_rr = np.load(tf_folder + "%s_pow_ent_left_dSPM_%s_%s" % (
                subject, cor, p) + "_Brodmann.17-rh_target.npy")
            ent_lc_lr = ent_lc_lr.mean(axis=0).mean(axis=0)
            ent_lc_rr = ent_lc_rr.mean(axis=0).mean(axis=0)

            ali_ent_left = ((ent_lc_lr - ent_lc_rr) / (ent_lc_lr + ent_lc_rr))

            ent_rc_lr = np.load(tf_folder + "%s_pow_ent_right_dSPM_%s_%s" % (
                subject, cor, p) + "_Brodmann.17-lh_target.npy")
            ent_rc_rr = np.load(tf_folder + "%s_pow_ent_right_dSPM_%s_%s" % (
                subject, cor, p) + "_Brodmann.17-rh_target.npy")
            ent_rc_lr = ent_rc_lr.mean(axis=0).mean(axis=0)
            ent_rc_rr = ent_rc_rr.mean(axis=0).mean(axis=0)

            ali_ent_right = ((ent_rc_lr - ent_rc_rr) / (ent_rc_lr + ent_rc_rr))

            # ent right
            row = pd.DataFrame([{
                "subject": subject,
                "type": "ent",
                "side": "right",
                "correct": cor,
                "phase": p,
                "ALI_pow": ali_ent_right[from_time:to_time].mean()
            }])
            df = df.append(row, ignore_index=True)
            # ent left
            row = pd.DataFrame([{
                "subject": subject,
                "type": "ent",
                "side": "left",
                "correct": cor,
                "phase": p,
                "ALI_pow": ali_ent_left[from_time:to_time].mean()
            }])
            df = df.append(row, ignore_index=True)
            # ctl right
            row = pd.DataFrame([{
                "subject": subject,
                "type": "ctl",
                "side": "right",
                "correct": cor,
                "phase": p,
                "ALI_pow": ali_ctl_right[from_time:to_time].mean()
            }])
            df = df.append(row, ignore_index=True)
            # ctl left
            row = pd.DataFrame([{
                "subject": subject,
                "type": "ctl",
                "side": "left",
                "correct": cor,
                "phase": p,
                "ALI_pow": ali_ctl_left[from_time:to_time].mean()
            }])
            df = df.append(row, ignore_index=True)

df.to_csv(
    result_dir + "alpha_ali_mean_data_extracted_phase_target.csv", index=False)
