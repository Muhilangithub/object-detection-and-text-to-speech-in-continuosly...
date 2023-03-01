import cv2
import numpy as np
import pyttsx3
import time

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Load the image

##############################################################################################################################

engine = pyttsx3.init()

# Set the speed and volume of the voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 120) # Speed in words per minute
engine.setProperty('volume', 1) # Volume (0 to 1)

##############################################################################################################################
# Create a blob from the image and pass it to the network




# Define the object class names
classNames = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
              'sofa', 'train', 'tvmonitor']


list_reg = []
no_not_detected = 0
# Loop over the detections and draw bounding boxes around the objects
while True:
    #image = cv2.imread('image.jpg')
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    cap.release()
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    entry_status = False
    print(no_not_detected)
    if no_not_detected < 5:
        
        for i in range(detections.shape[2]):            
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                entry_status = True
                class_id = int(detections[0, 0, i, 1])
                class_name = classNames[class_id]
                #box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                #(startX, startY, endX, endY) = box.astype("int")
                confi=confidence*100
                confi_int =int(confi)
                label = f"{class_name} {confi_int}%"    
                '''#{confidence:.2f}'''
                list_reg.append(label)
                no_not_detected = 0
    else:
        engine.say("shutdowning object detection mode")
        engine.runAndWait()
        break

    if entry_status == False:
        no_not_detected +=1
        time.sleep(4)
        print("breaked")
        continue
            
    else:
        for label in list_reg:
            engine.say(label)
            time.sleep(0.5)
        engine.runAndWait()
        print(list_reg)
        list_reg = []
        print("get empty",list_reg)

                #cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                #y = startY - 15 if startY - 15 > 15 else startY + 15
                #cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
################################################################################################################################        
       # Show the output image
       #cv2.imshow("Output", image)
       #cv2.waitKey(0)
       #cv2.destroyAllWindows()
    
        
#################################################################################################################################
