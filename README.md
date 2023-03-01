# object-detection-and-text-to-speech-in-continuosly...
this code is detect the object that in front of camera(only trained 18 objects) and convert it into speech using python.
#######################################################################################################################################################################


1.try3.py is python file which recognize the object from you web cam within the the list which the model is trained and convert it using pyttsx3

2.requirements for this project
     (+)import cv2
     (+)import numpy as np
     (+)import pyttsx3
     (+)import time
     (+)import sys
 3.there is also kivy app file is there (new_apppppp.py) which do the same with additionally which terminate the code itself when more than 5 trials it doesn't detect nothing, 4 sec each trails.
 
 4.to run those codes the below conditions must be passed
     (+)MobileNetSSD_deploy.caffemodel and MobileNetSSD_deploy.prototxt.txt and try3.py or new_apppppp.py must be placed same directory.
     (+)all the requirements must be passed 
