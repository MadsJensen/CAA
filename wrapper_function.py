"""
Doc string here.

@author mje
@email: mads [] cnru.dk

"""
import sys
import subprocess

subjects_select = ["0005", "0006", "0007", "0008", "0009", "0010",
                   "0011", "0015", "0016", "0017", "0020", "0021",
                   "0022", "0024", "0025"]

if len(sys.argv) == 3:
    cpu_number = sys.argv[2]
else:
    cpu_number = 1


for subject in subjects_select:
    submit_cmd = 'submit_to_cluster \"python %s %s\"' % (sys.argv[1], subject)
    # print(submit_cmd)
    subprocess.call([submit_cmd])
