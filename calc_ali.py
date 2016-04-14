import numpy as np

from my_settings import *


def calc_ali(subject, method="dSPM"):
    """
    Params
    ------
    subject : the subject number
    method : str
        The method used for inverse model.
    """

    ctl_left_roi_left_cue = np.load(tf_folder +
                                    "%s_pow_ctl_left_%s_Brodmann.17-lh.npy"
                                    % (subject, method))
    ctl_right_roi_left_cue = np.load(tf_folder +
                                     "%s_pow_ctl_left_%s_Brodmann.17-rh.npy"
                                     % (subject, method))
    ctl_left_roi_right_cue = np.load(tf_folder +
                                     "%s_pow_ctl_right_%s_Brodmann.17-lh.npy"
                                     % (subject, method))
    ctl_right_roi_right_cue = np.load(tf_folder +
                                      "%s_pow_ctl_right_%s_Brodmann.17-rh.npy"
                                      % (subject, method))

    ent_left_roi_left_cue = np.load(tf_folder +
                                    "%s_pow_ent_left_%s_Brodmann.17-lh.npy"
                                    % (subject, method))
    ent_right_roi_left_cue = np.load(tf_folder +
                                     "%s_pow_ent_left_%s_Brodmann.17-rh.npy"
                                     % (subject, method))
    ent_left_roi_right_cue = np.load(tf_folder +
                                     "%s_pow_ent_right_%s_Brodmann.17-lh.npy"
                                     % (subject, method))
    ent_right_roi_right_cue = np.load(tf_folder +
                                      "%s_pow_ent_right_%s_Brodmann.17-rh.npy"
                                      % (subject, method))

    ali_left_cue_ctl = ((ctl_left_roi_left_cue.mean(axis=0).mean(axis=0) -
                         ctl_right_roi_left_cue.mean(axis=0).mean(axis=0)) /
                        (ctl_left_roi_left_cue.mean(axis=0).mean(axis=0) +
                         ctl_right_roi_left_cue.mean(axis=0).mean(axis=0)))

    ali_right_cue_ctl = ((ctl_left_roi_right_cue.mean(axis=0).mean(axis=0) -
                          ctl_right_roi_right_cue.mean(axis=0).mean(axis=0)) /
                         (ctl_left_roi_right_cue.mean(axis=0).mean(axis=0) +
                          ctl_right_roi_right_cue.mean(axis=0).mean(axis=0)))

    ali_left_cue_ent = ((ent_left_roi_left_cue.mean(axis=0).mean(axis=0) -
                         ent_right_roi_left_cue.mean(axis=0).mean(axis=0)) /
                        (ent_left_roi_left_cue.mean(axis=0).mean(axis=0) +
                         ent_right_roi_left_cue.mean(axis=0).mean(axis=0)))

    ali_right_cue_ent = ((ent_left_roi_right_cue.mean(axis=0).mean(axis=0) -
                          ent_right_roi_right_cue.mean(axis=0).mean(axis=0)) /
                         (ent_left_roi_right_cue.mean(axis=0).mean(axis=0) +
                          ent_right_roi_right_cue.mean(axis=0).mean(axis=0)))

    return (ali_left_cue_ctl, ali_right_cue_ctl,
            ali_left_cue_ent, ali_right_cue_ent)
