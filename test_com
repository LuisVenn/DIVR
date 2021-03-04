from paramiko import SSHClient
import subprocess

client = SSHClient()
client.load_system_host_keys()
client.connect('192.168.0.11',22,'pi','divr2021')

return01 = subprocess.Popen(['python3','video.py'], cwd='/home/pi/Desktop') 
stdin,stdout,stderr = client.exec_command('cd Desktop;python3 video.py') 


client.close()

print('Feito')
