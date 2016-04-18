"""
Doc string here.

@author mje
@email: mads [] cnru.dk

"""
import subprocess

cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"

subjects_select = ["0005", "0006", "0007", "0008", "0009", "0010",
                   "0011", "0015", "0016", "0017", "0020", "0021",
                   "0022", "0024", "0025"]
                   
for subject in subjects_select:
    submit_cmd = "mne_setup_source_space --subject %s --surf --oct 6" % subject
    subprocess.call([cmd, "1", submit_cmd])
