import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Compute a label/ROI based on the peak power between 80 and 120 ms.
# The label bankssts-lh is used for the comparison.
tmin, tmax = 0.1, 0.7


def make_functional_label(epochs, inv, side, tmin=0.1, tmax=0.7,
                          save=True):
    """
    Params
    ------
    epochs : Epochs
        The epochs to use for the functional label.
    side : str
        String with the cued side.
    tmin : float
        start time.
    tmax : float
        end time.
    save : bool
        To save the label or not.

    Returns
    -------
    label_rh : freesurfer label
        label for the right hemisphere
    label_lh : freesurfer label
        label for the left hemisphere
    """

    # Load data
    subject_name = epochs.info["subject_info"]["last_name"]
    evoked = epochs[side].average()
    src = inv['src']  # get the source space

    labels = mne.read_labels_from_annot(subject_name,
                                        parc='PALS_B12_Brodmann',
                                        regexp="Bro",
                                        subjects_dir=subjects_dir)
    label_lh, labels_rh = labels[6], labels[7]

    # Compute inverse solution
    stc = apply_inverse(evoked, inv, lambda2, method,
                        pick_ori='normal')

    # Make an STC in the time interval of interest and take the mean
    stc_mean = stc.copy().crop(tmin, tmax).mean()

    # use the stc_mean to generate a functional label
    # region growing is halted at 60% of the peak value within the
    # anatomical label / ROI specified by aparc_label_name

    # calc lh label
    stc_mean_label = stc_mean.in_label(label_lh)
    data = np.abs(stc_mean_label.data)
    stc_mean_label.data[data < 0.65 * np.max(data)] = 0.

    func_labels_lh, _ = mne.stc_to_label(stc_mean_label,
                                         src=src,
                                         smooth=True,
                                         subjects_dir=subjects_dir,
                                         connected=True)
    # take first as func_labels are ordered based on maximum values in stc
    func_label_lh = func_labels_lh[0]

    # calc rh label
    stc_mean_label = stc_mean.in_label(label_lh)
    data = np.abs(stc_mean_label.data)
    stc_mean_label.data[data < 0.65 * np.max(data)] = 0.

    _, func_labels_rh = mne.stc_to_label(stc_mean_label,
                                         src=src,
                                         smooth=True,
                                         subjects_dir=subjects_dir,
                                         connected=True)
    # take first as func_labels are ordered based on maximum values in stc
    func_label_lh = func_labels_lh[0]




    if save:
        func_label_lh.save(mne_folder + "%s-func_label_cond-%s_lh"
                           % (subject_name, side))
        func_label_rh.save(mne_folder + "%s-func_label_cond-%s_rh"
                           % (subject_name, side))
