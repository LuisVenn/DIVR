from picamera import PiCamera
import time

camera = PiCamera()
camera.rotation = 180
camera.resolution = (1640,1232)
time.sleep(2)

ID="01"
variable =("Photo_Cam"+ID+time.strftime("_%H-%M_%d-%b-%y"))

camera.capture("/home/pi/Desktop/Testes/" + variable + ".jpg")

