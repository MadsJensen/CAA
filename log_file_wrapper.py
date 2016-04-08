# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 13:43:41 2016

@author: mje
"""
import pandas as pd
import mne

subjects = ["0005",  "0006",  "0007",  "0008",  "0009", "0010", "0011",
            "0015", "0016", "0017", "0020", "0021", "0022", "0024", "0025"]

for subject in subjects:
    raw = mne.io.Raw(maxfiltered_folder + "%s_data_mc_raw_tsss.fif" % subject)
    events = mne.find_events(raw, min_duration=0.01)
    mne.write_events(log_folder + "%s-eve.fif" % subject, events)


for subject in subjects:
    df = make_log_file(subject)
    df.to_csv(log_folder + "%s_log_file.csv" %subject, index=False)

results = pd.DataFrame()
for subject in subjects:
    df = pd.read_csv(log_folder + "%s_log_file.csv" %subject)
    df["subject"] = subject

    results = results.append(df, ignore_index=True)
