import numpy as np

from my_settings import *

ctl_left_roi_left_cue = np.load(tf_folder +
                                "%s_pow_ctl_left_Brodmann.17-lh.npy"
                                % (subject))
ctl_right_roi_left_cue = np.load(tf_folder +
                                 "%s_pow_ctl_left_Brodmann.17-rh.npy"
                                 % (subject))
ctl_left_roi_right_cue = np.load(tf_folder +
                                 "%s_pow_ctl_right_Brodmann.17-lh.npy"
                                 % (subject))
ctl_right_roi_right_cue = np.load(tf_folder +
                                  "%s_pow_ctl_right_Brodmann.17-rh.npy"
                                  % (subject))

ALI_left_cue_ctl = ((ctl_left_roi_left_cue.mean(axis=0).mean(axis=0) -
                     ctl_right_roi_left_cue.mean(axis=0).mean(axis=0)) /
                    (ctl_left_roi_left_cue.mean(axis=0).mean(axis=0) +
                     ctl_right_roi_left_cue.mean(axis=0).mean(axis=0)))

ALI_right_cue_ctl = ((ctl_left_roi_right_cue.mean(axis=0).mean(axis=0) -
                      ctl_right_roi_right_cue.mean(axis=0).mean(axis=0)) /
                     (ctl_left_roi_right_cue.mean(axis=0).mean(axis=0) +
                      ctl_right_roi_right_cue.mean(axis=0).mean(axis=0)))
