# coding=utf-8
"""
This is a group of function to be used on TF data.

@author: mje
@email: mads [] cnru.dk
"""

from my_settings import (epochs_folder)
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt

selection_name = "Left-occipital"
subject = sys.argv[1]


def find_channel_index(subject, selection=selection_name):
    epochs = mne.read_epochs(
        epochs_folder + "%s_trial_start-epo.fif" % subject, preload=False)
    selection = mne.read_selection(selection_name)
    selection = [f.split()[0] + f.split()[1] for f in selection]
    picks = mne.pick_types(epochs.info, meg='grad', eeg=False, eog=False,
                           stim=False, exclude=[], selection=selection)

    return picks


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
    ctl_left_roi_left_cue =\
        mne.read_source_estimate(tf_folder +
                                 "BP_%s_ctl_left_OCCIPITAL_lh_dSPM"
                                 % (subject))
    ctl_right_roi_left_cue =\
        mne.read_source_estimate(tf_folder +
                                 "BP_%s_ctl_left_OCCIPITAL_rh_dSPM"
                                 % (subject))
    ctl_left_roi_right_cue =\
        mne.read_source_estimate(tf_folder +
                                 "BP_%s_ctl_right_OCCIPITAL_lh_dSPM"
                                 % (subject))
    ctl_right_roi_right_cue =\
        mne.read_source_estimate(tf_folder +
                                 "BP_%s_ctl_right_OCCIPITAL_rh_dSPM"
                                 % (subject))

    ALI_left_cue_ctl = ((ctl_left_roi_left_cue.data.mean(axis=0) -
                         ctl_right_roi_left_cue.data.mean(axis=0)) /
                        (ctl_left_roi_left_cue.data.mean(axis=0) +
                         ctl_right_roi_left_cue.data.mean(axis=0)))

    ALI_right_cue_ctl = ((ctl_left_roi_right_cue.data.mean(axis=0) -
                          ctl_right_roi_right_cue.data.mean(axis=0)) /
                         (ctl_left_roi_right_cue.data.mean(axis=0) +
                          ctl_right_roi_right_cue.data.mean(axis=0)))

    ent_left_roi_left_cue =\
        mne.read_source_estimate(tf_folder +
                                 "BP_%s_ent_left_OCCIPITAL_lh_dSPM"
                                 % (subject))
    ent_right_roi_left_cue =\
        mne.read_source_estimate(tf_folder +
                                 "BP_%s_ent_left_OCCIPITAL_rh_dSPM"
                                 % (subject))
    ent_left_roi_right_cue =\
        mne.read_source_estimate(tf_folder +
                                 "BP_%s_ent_right_OCCIPITAL_lh_dSPM"
                                 % (subject))
    ent_right_roi_right_cue =\
        mne.read_source_estimate(tf_folder +
                                 "BP_%s_ent_right_OCCIPITAL_rh_dSPM"
                                 % (subject))

    ALI_left_cue_ent = ((ent_left_roi_left_cue.data.mean(axis=0) -
                         ent_right_roi_left_cue.data.mean(axis=0)) /
                        (ent_left_roi_left_cue.data.mean(axis=0) +
                         ent_right_roi_left_cue.data.mean(axis=0)))

    ALI_right_cue_ent = ((ent_left_roi_right_cue.data.mean(axis=0) -
                          ent_right_roi_right_cue.data.mean(axis=0)) /
                         (ent_left_roi_right_cue.data.mean(axis=0) +
                          ent_right_roi_right_cue.data.mean(axis=0)))

    if show_plot:
        times = ent_left_roi_left_cue.times
        plt.figure()
        plt.plot(times, ALI_left_cue_ctl, 'r', label="ALI Left cue control")
        plt.plot(times, ALI_left_cue_ent, 'b', label="ALI Left ent control")
        plt.plot(times, ALI_right_cue_ctl, 'g', label="ALI Right cue control")
        plt.plot(times, ALI_right_cue_ent, 'm', label="ALI Right ent control")
        plt.legend()
        plt.title("ALI curves for subject: %s" % subject)
        plt.show()

    return (ALI_left_cue_ctl, ALI_right_cue_ctl,
            ALI_left_cue_ent, ALI_right_cue_ent)


