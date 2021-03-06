import mne
from my_settings import *
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
n_jobs = 3

for subject in [subjects_select[-1]]:
    raw = mne.io.Raw(save_folder + "%s_filtered_ica_mc_raw_tsss.fif" % subject,
                     preload=True)
    raw.resample(250, n_jobs=n_jobs, verbose=True)
    raw.filter(8, 12, n_jobs=n_jobs, verbose=True)

    include = []
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False,
                           include=include, exclude='bads')
    raw.apply_hilbert(picks, n_jobs=n_jobs, verbose=True)
    raw.save(save_folder + "%s_hilbert_ica_mc_raw_tsss.fif" % subject,
             overwrite=True)

    tmin, tmax = -0.5, 1.5  # Epoch time

    # All the behavioral results
    results = pd.read_csv(log_folder + "results_all.csv")
    # select only the relevant subject
    log_tmp = results[results.subject == int(subject)].reset_index()

    raw.del_proj(0)

    # Select events to extract epochs from.
    event_id = {"all_trials": 99}

    #   Setup for reading the raw data
    events = mne.find_events(raw, min_duration=0.015)
    events = mne.event.merge_events(events, [1, 2, 4, 8], 99,
                                    replace_events=True)

    event_id = {}
    epoch_ids = []
    for i, row in log_tmp.iterrows():
        if row.condition_type == "ctl":
            epoch_name = "ctl"
            epoch_id = "1"
        elif row.condition_type == "ent":
            epoch_name = "ent"
            epoch_id = "2"

        if row.condition_side == "left":
            epoch_name = epoch_name + "/" + "left"
            epoch_id = epoch_id + "1"
        elif row.condition_side == "right":
            epoch_name = epoch_name + "/" + "right"
            epoch_id = epoch_id + "0"

        if row.congruent is True:
            epoch_name = epoch_name + "/" + "cong"
            epoch_id = epoch_id + "1"
        elif row.congruent is False:
            epoch_name = epoch_name + "/" + "incong"
            epoch_id = epoch_id + "0"

        if row.correct is True:
            epoch_name = epoch_name + "/" + "correct"
            epoch_id = epoch_id + "1"
        elif row.correct is False:
            epoch_name = epoch_name + "/" + "incorrect"
            epoch_id = epoch_id + "0"

        if row.in_phase is True:
            epoch_name = epoch_name + "/" + "in_phase"
            epoch_id = epoch_id + "1"
        elif row.in_phase is False:
            epoch_name = epoch_name + "/" + "out_phase"
            epoch_id = epoch_id + "0"

        epoch_name = epoch_name + "/" + str(row.PAS)
        epoch_id = epoch_id + str(row.PAS)
        epoch_ids.append(int(epoch_id))

        if epoch_name is not event_id:
            event_id[str(epoch_name)] = int(epoch_id)

    idx = np.arange(0, len(events), 4)
    for i in range(len(events[idx])):
        events[idx[i]][2] = epoch_ids[i]

    # picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False,
    # eog=False,
    #                        include=include, exclude='bads')
    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, -0.2), reject=reject_params,
                        add_eeg_ref=True,
                        preload=False)
    epochs.drop_bad_epochs(reject_params)

    fig = epochs.plot_drop_log(subject=subject, show=False)
    fig.savefig(epochs_folder + "pics/hilbert_%s_drop_log.png" % subject)

    epochs.save(epochs_folder + "%s_hilbert_trial_start-epo.fif" % subject)
