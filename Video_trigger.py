from paramiko import SSHClient
import subprocess

client2 = SSHClient()
client2.load_system_host_keys()
client2.connect('192.168.43.168',22,'pi','divr2021')
print('Connected to 1')

client3 = SSHClient()
client3.load_system_host_keys()
client3.connect('192.168.43.162',22,'pi','divr2021')
print('Connected to 3')

stdin,stdout,stderr = client2.exec_command('cd Desktop;python3 foto.py') 
stdin,stdout,stderr = client3.exec_command('cd Desktop;python3 foto.py') 

client2.close()
client3.close()

print('Feito')
