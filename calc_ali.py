# coding=utf-8
"""
This is a group of function to be used on TF data.

@author: mje
@email: mads [] cnru.dk
"""

from my_settings import (epochs_folder, tf_folder)
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt

subject = sys.argv[1]

epochs = mne.read_epochs(
    epochs_folder + "%s_trial_start-epo.fif" % subject, preload=False)
selection = mne.read_selection("Left-occipital")
selection = [f.replace(' ', '') for f in selection]
left_idx = mne.pick_types(
    epochs.info,
    meg='grad',
    eeg=False,
    eog=False,
    stim=False,
    exclude=[],
    selection=selection)

selection = mne.read_selection("Right-occipital")
selection = [f.replace(' ', '') for f in selection]
right_idx = mne.pick_types(
    epochs.info,
    meg='grad',
    eeg=False,
    eog=False,
    stim=False,
    exclude=[],
    selection=selection)


def calc_ALI(subject, show_plot=False):
    """Function calculates the alpha lateralization index (ALI).

    The alpha lateralization index (ALI) is based on:
    Huurne, N. ter, Onnink, M., Kan, C., Franke, B., Buitelaar, J.,
    & Jensen, O. (2013).
    Parameters
    ----------
    subject : string
        The name of the subject to calculate ALI for.
    show_plot : bool
        Whether to plot the data or not.

    RETURNS
    -------
    ali_left : the ALI for the left cue
    ali_right : the ALI for the right cue
    """
    ctl_left = np.load(tf_folder + "%s_ctl_left-4-tfr.npy" % (subject))
    ctl_right = np.load(tf_folder + "%s_ctl_right-4-tfr.npy" % (subject))
    ent_left = np.load(tf_folder + "%s_ent_left-4-tfr.npy" % (subject))
    ent_right = np.load(tf_folder + "%s_ent_right-4-tfr.npy" % (subject))

    ALI_left_cue_ctl = ((ctl_left[:, left_idx, :, :].mean(axis=0).mean(
        axis=0) - ctl_left[:, right_idx, :, :].mean(axis=0).mean(axis=0)) / (
            ctl_left[:, left_idx, :, :].mean(axis=0).mean(axis=0) +
            ctl_left[:, right_idx, :, :].mean(axis=0).mean(axis=0)))

    ALI_right_cue_ctl = ((ctl_right[:, left_idx, :, :].mean(axis=0).mean(
        axis=0) - ctl_right[:, right_idx, :, :].mean(axis=0).mean(axis=0)) / (
            ctl_right[:, left_idx, :, :].mean(axis=0).mean(axis=0) +
            ctl_right[:, right_idx, :, :].mean(axis=0).mean(axis=0)))

    ALI_left_cue_ent = ((ent_left[:, left_idx, :, :].mean(axis=0).mean(
        axis=0) - ent_left[:, right_idx, :, :].mean(axis=0).mean(axis=0)) / (
            ent_left[:, left_idx, :, :].mean(axis=0).mean(axis=0) +
            ent_left[:, right_idx, :, :].mean(axis=0).mean(axis=0)))

    ALI_right_cue_ent = ((ent_right[:, left_idx, :, :].mean(axis=0).mean(
        axis=0) - ent_right[:, right_idx, :, :].mean(axis=0).mean(axis=0)) / (
            ent_right[:, left_idx, :, :].mean(axis=0).mean(axis=0) +
            ent_right[:, right_idx, :, :].mean(axis=0).mean(axis=0)))

    if show_plot:
        times = epochs.times
        plt.figure()
        plt.plot(times, ALI_left_cue_ctl, 'r', label="ALI Left cue control")
        plt.plot(times, ALI_left_cue_ent, 'b', label="ALI Left ent control")
        plt.plot(times, ALI_right_cue_ctl, 'g', label="ALI Right cue control")
        plt.plot(times, ALI_right_cue_ent, 'm', label="ALI Right ent control")
        plt.legend()
        plt.title("ALI curves for subject: %s" % subject)
        plt.show()

    return (ALI_left_cue_ctl.mean(axis=0), ALI_right_cue_ctl.mean(axis=0),
            ALI_left_cue_ent.mean(axis=0), ALI_right_cue_ent.mean(axis=0))



ctl_left_ali, ctl_right_ali, ent_left_ali, ent_right_ali = calc_ALI(subject)

data = np.vstack((ctl_left_ali, ctl_right_ali, ent_left_ali, ent_right_ali))
np.save(tf_folder + "%s_ali.npy" % subject, data)
