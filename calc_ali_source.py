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

    ALI_left_cue_ctl = (
        (ctl_left[:, left_idx, :, :].mean(axis=0).mean(axis=0) -
         ctl_left[:, right_idx, :, :].mean(axis=0).mean(axis=0)) /
        (ctl_left[:, left_idx, :, :].mean(axis=0).mean(axis=0) +
         ctl_left[:, right_idx, :, :].mean(axis=0).mean(axis=0)))

    ALI_right_cue_ctl = (
        (ctl_right[:, left_idx, :, :].mean(axis=0).mean(axis=0) -
         ctl_right[:, right_idx, :, :].mean(axis=0).mean(axis=0)) /
        (ctl_right[:, left_idx, :, :].mean(axis=0).mean(axis=0) +
         ctl_right[:, right_idx, :, :].mean(axis=0).mean(axis=0)))

    ALI_left_cue_ent = (
        (ent_left[:, left_idx, :, :].mean(axis=0).mean(axis=0) -
         ent_left[:, right_idx, :, :].mean(axis=0).mean(axis=0)) /
        (ent_left[:, left_idx, :, :].mean(axis=0).mean(axis=0) +
         ent_left[:, right_idx, :, :].mean(axis=0).mean(axis=0)))

    ALI_right_cue_ent = (
        (ent_right[:, left_idx, :, :].mean(axis=0).mean(axis=0) -
         ent_right[:, right_idx, :, :].mean(axis=0).mean(axis=0)) /
        (ent_right[:, left_idx, :, :].mean(axis=0).mean(axis=0) +
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


def calc_ALI_source(subject):
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
    ctl_right_rh = np.load(
        tf_folder +
        "%s_ctl_right_LOBE.OCCIPITAL-rh_MNE_source_power_snr_3.npy" %
        (subject))
    ctl_right_lh = np.load(
        tf_folder +
        "%s_ctl_right_LOBE.OCCIPITAL-lh_MNE_source_power_snr_3.npy" %
        (subject))
    ctl_left_rh = np.load(
        tf_folder + "%s_ctl_left_LOBE.OCCIPITAL-rh_MNE_source_power_snr_3.npy"
        % (subject))
    ctl_left_lh = np.load(
        tf_folder + "%s_ctl_left_LOBE.OCCIPITAL-lh_MNE_source_power_snr_3.npy"
        % (subject))

    ent_right_rh = np.load(
        tf_folder +
        "%s_ent_right_LOBE.OCCIPITAL-rh_MNE_source_power_snr_3.npy" %
        (subject))
    ent_right_lh = np.load(
        tf_folder +
        "%s_ent_right_LOBE.OCCIPITAL-lh_MNE_source_power_snr_3.npy" %
        (subject))
    ent_left_rh = np.load(
        tf_folder + "%s_ent_left_LOBE.OCCIPITAL-rh_MNE_source_power_snr_3.npy"
        % (subject))
    ent_left_lh = np.load(
        tf_folder + "%s_ent_left_LOBE.OCCIPITAL-lh_MNE_source_power_snr_3.npy"
        % (subject))

    # Select top 90% sources
    ctl_left_lh = np.percentile(ctl_left_lh, 90, axis=0)
    ctl_left_rh = np.percentile(ctl_left_rh, 90, axis=0)
    ctl_right_lh = np.percentile(ctl_right_lh, 90, axis=0)
    ctl_right_rh = np.percentile(ctl_right_rh, 90, axis=0)

    ent_left_lh = np.percentile(ent_left_lh, 90, axis=0)
    ent_left_rh = np.percentile(ent_left_rh, 90, axis=0)
    ent_right_lh = np.percentile(ent_right_lh, 90, axis=0)
    ent_right_rh = np.percentile(ent_right_rh, 90, axis=0)

    ALI_left_cue_ctl = (
        (ctl_left_lh - ctl_left_rh) / (ctl_left_lh + ctl_left_rh))

    ALI_right_cue_ctl = (
        (ctl_right_lh - ctl_right_rh) / (ctl_right_lh + ctl_right_lh))

    ALI_left_cue_ent = (
        (ent_left_lh - ent_left_rh) / (ent_left_lh + ent_left_rh))

    ALI_right_cue_ent = (
        (ent_right_lh - ent_right_rh) / (ent_right_lh + ent_right_lh))

    return (ALI_left_cue_ctl, ALI_right_cue_ctl, ALI_left_cue_ent,
            ALI_right_cue_ent)


def calc_ALI_ITC_source(subject):
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
    ctl_right_rh = np.load(
        tf_folder +
        "%s_ctl_right_LOBE.OCCIPITAL-rh_MNE_source_itc_snr_3.npy" % (subject))
    ctl_right_lh = np.load(
        tf_folder +
        "%s_ctl_right_LOBE.OCCIPITAL-lh_MNE_source_itc_snr_3.npy" % (subject))
    ctl_left_rh = np.load(
        tf_folder + "%s_ctl_left_LOBE.OCCIPITAL-rh_MNE_source_itc_snr_3.npy" %
        (subject))
    ctl_left_lh = np.load(
        tf_folder + "%s_ctl_left_LOBE.OCCIPITAL-lh_MNE_source_itc_snr_3.npy" %
        (subject))

    ent_right_rh = np.load(
        tf_folder +
        "%s_ent_right_LOBE.OCCIPITAL-rh_MNE_source_itc_snr_3.npy" % (subject))
    ent_right_lh = np.load(
        tf_folder +
        "%s_ent_right_LOBE.OCCIPITAL-lh_MNE_source_itc_snr_3.npy" % (subject))
    ent_left_rh = np.load(
        tf_folder + "%s_ent_left_LOBE.OCCIPITAL-rh_MNE_source_itc_snr_3.npy" %
        (subject))
    ent_left_lh = np.load(
        tf_folder + "%s_ent_left_LOBE.OCCIPITAL-lh_MNE_source_itc_snr_3.npy" %
        (subject))

    # Select top 90% sources
    ctl_left_lh = np.percentile(ctl_left_lh, 90, axis=0)
    ctl_left_rh = np.percentile(ctl_left_rh, 90, axis=0)
    ctl_right_lh = np.percentile(ctl_right_lh, 90, axis=0)
    ctl_right_rh = np.percentile(ctl_right_rh, 90, axis=0)

    ent_left_lh = np.percentile(ent_left_lh, 90, axis=0)
    ent_left_rh = np.percentile(ent_left_rh, 90, axis=0)
    ent_right_lh = np.percentile(ent_right_lh, 90, axis=0)
    ent_right_rh = np.percentile(ent_right_rh, 90, axis=0)

    ALI_left_cue_ctl = (
        (ctl_left_lh - ctl_left_rh) / (ctl_left_lh + ctl_left_rh))

    ALI_right_cue_ctl = (
        (ctl_right_lh - ctl_right_rh) / (ctl_right_lh + ctl_right_lh))

    ALI_left_cue_ent = (
        (ent_left_lh - ent_left_rh) / (ent_left_lh + ent_left_rh))

    ALI_right_cue_ent = (
        (ent_right_lh - ent_right_rh) / (ent_right_lh + ent_right_lh))

    return (ALI_left_cue_ctl, ALI_right_cue_ctl, ALI_left_cue_ent,
            ALI_right_cue_ent)


# Calc ali for ITC
ctl_left_ali, ctl_right_ali, ent_left_ali, ent_right_ali = calc_ALI_ITC_source(
    subject)
data = np.vstack((ctl_left_ali, ctl_right_ali, ent_left_ali, ent_right_ali))
np.save(tf_folder + "%s_ali_itc_source_LOBE.OCCIPITAL.npy" % subject, data)

# Calc ali for power
ctl_left_ali, ctl_right_ali, ent_left_ali, ent_right_ali = calc_ALI_source(
    subject)
data = np.vstack((ctl_left_ali, ctl_right_ali, ent_left_ali, ent_right_ali))
np.save(tf_folder + "%s_ali_pow_source_LOBE.OCCIPITAL.npy" % subject, data)
