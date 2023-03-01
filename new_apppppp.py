import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window

import cv2
import numpy as np
import pyttsx3
import time
import sys 
Window.size = (375,667)
class MyApp(App):
    counter = 0
    def build(self):
        # create a box layout
        layout = BoxLayout(orientation='vertical')

        # create two buttons
        button1 = Button(text='Detect Object', size_hint=(1, None), height=100)
        button2 = Button(text='sos soon...', size_hint=(1, None), height=100)

        # add the buttons to the layout
        layout.add_widget(button1)
        layout.add_widget(button2)

        button1.bind(on_press=self.detect_object)
        return layout
    
    def detect_object(self,layout):
        if self.counter == 0:
            net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id)
            engine.setProperty('rate', 120) # Speed in words per minute
            engine.setProperty('volume', 1)

            classNames = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                'sofa', 'train', 'tvmonitor']


            list_reg = []
            no_not_detected = 0
            while True:
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
                            confi=confidence*100
                            confi_int =int(confi)
                            label = f"{class_name} {confi_int}%"    
                            list_reg.append(label)
                            no_not_detected = 0
                else:
                    engine.say("shutdowning object detection mode")
                    engine.runAndWait()
                    sys.exit()
                    #break
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
        elif self.counter == 1:
            print("hi")
            self.counter = -1


        self.counter += 1


        

if __name__ == '__main__':
    MyApp().run()

