import subprocess

list_files = subprocess.run(["ls","-l"], cwd='/home/pi/Desktop')
print("the exit code was: %d" % list_files.returncode)
