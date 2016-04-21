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
                                    "%s_ctl_left_Brodmann.17-lh_%s-tfr.npy"
                                    % (subject, method))
    ctl_right_roi_left_cue = np.load(tf_folder +
                                     "%s_ctl_left_Brodmann.17-rh_%s-tfr.npy"
                                     % (subject, method))
    ctl_left_roi_right_cue = np.load(tf_folder +
                                     "%s_ctl_right_Brodmann.17-lh_%s-tfr.npy"
                                     % (subject, method))
    ctl_right_roi_right_cue = np.load(tf_folder +
                                      "%s_ctl_right_Brodmann.17-rh_%s-tfr.npy"
                                      % (subject, method))

    ent_left_roi_left_cue = np.load(tf_folder +
                                    "%s_ent_left_Brodmann.17-lh_%s-tfr.npy"
                                    % (subject, method))
    ent_right_roi_left_cue = np.load(tf_folder +
                                     "%s_ent_left_Brodmann.17-rh_%s-tfr.npy"
                                     % (subject, method))
    ent_left_roi_right_cue = np.load(tf_folder +
                                     "%s_ent_left_Brodmann.17-lh_%s-tfr.npy"
                                     % (subject, method))
    ent_right_roi_right_cue = np.load(tf_folder +
                                      "%s_ent_right_Brodmann.17-rh_%s-tfr.npy"
                                      % (subject, method))

    ali_left_cue_ctl =\
        ((np.mean(np.mean(np.abs(ctl_left_roi_left_cue)**2, axis=0), axis=0) -
          np.mean(np.mean(np.abs(ctl_right_roi_left_cue)**2, axis=0),
                  axis=0)) /
         (np.mean(np.mean(np.abs(ctl_left_roi_left_cue)**2, axis=0), axis=0) +
          np.mean(np.mean(np.abs(ctl_right_roi_left_cue)**2, axis=0), axis=0)))

    ali_right_cue_ctl =\
        ((np.mean(np.mean(np.abs(ctl_left_roi_right_cue)**2, axis=0), axis=0) -
          np.mean(np.mean(np.abs(ctl_right_roi_right_cue)**2, axis=0),
                  axis=0)) /
         (np.mean(np.mean(np.abs(ctl_left_roi_right_cue)**2, axis=0), axis=0) +
          np.mean(np.mean(np.abs(ctl_right_roi_right_cue)**2, axis=0),
                  axis=0)))

    ali_left_cue_ent =\
        ((np.mean(np.mean(np.abs(ent_left_roi_left_cue)**2, axis=0), axis=0) -
          np.mean(np.mean(np.abs(ent_right_roi_left_cue)**2, axis=0),
                  axis=0)) /
         (np.mean(np.mean(np.abs(ent_left_roi_left_cue)**2, axis=0), axis=0) +
          np.mean(np.mean(np.abs(ent_right_roi_left_cue)**2, axis=0), axis=0)))

    ali_right_cue_ent =\
        ((np.mean(np.mean(np.abs(ent_left_roi_right_cue)**2, axis=0), axis=0) -
          np.mean(np.mean(np.abs(ent_right_roi_right_cue)**2, axis=0),
                  axis=0)) /
         (np.mean(np.mean(np.abs(ent_left_roi_right_cue)**2, axis=0), axis=0) +
          np.mean(np.mean(np.abs(ent_right_roi_right_cue)**2, axis=0),
                  axis=0)))

    return (ali_left_cue_ctl, ali_right_cue_ctl,
            ali_left_cue_ent, ali_right_cue_ent)
