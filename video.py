############ MASTER CAM01 ##########

#Import Librarys
from picamera import PiCamera
import time
from time import sleep

#Camera Inicialization
camera = PiCamera()
camera.rotation = 180
camera.resolution = (1640,1232)

#File Name Definition
ID="01"
variable =("Cam"+ID+time.strftime("_%H-%M_%d-%b-%y"))

#Recording
camera.start_recording("/home/pi/Desktop/Testes/" + variable + ".h264")
sleep(5)
camera.stop_recording()
