"""
Doc string here.

@author mje
@email: mads [] cnru.dk

"""
import sys
import subprocess

cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"

submit_cmd = "python %s %s" % (sys.argv[1], sys.argv[2])
subprocess.call([cmd, "4", submit_cmd])
