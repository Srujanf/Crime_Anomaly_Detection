import os

import numpy as np
import matplotlib.pyplot as plt

import cv2
from tkinter import filedialog
import tensorflow as tf

model = tf.keras.models.load_model(r"../Models/model_x-y-22.h5", compile=False)

def setStateNormal(pred):
    ret = None
    if pred>0.5:
        ret = "Crime"
    else:
        ret = "Normal"
        
    return f"{ret} - {round(pred*100, 4)}%"


def ShowPredictionss(buffer_size:int=90):

    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    
    buffer = []
    prev_pred = "State: NaN"
    
    color = {
        1: (255, 0, 0), # blue 
        2: (0, 255, 0), # green
        3: (0, 0, 255), # red
        4: (106, 0, 0), # dark blue
        5: (0, 97, 0),  # dark green
        6: (0, 0, 96)   # dark red
    }
    
    multiplier=1
    outSize=(320*multiplier, 240*multiplier)

    cur_color = color[2]

    cap = cv2.VideoCapture(file_path)

    if cap.isOpened()==False:
        print("Either file not found or wrong codec used")

    while cap.isOpened():

        ret, frame = cap.read()

        if ret==True:

            # Make a copy of your frame and resize it to pass it to model
            out = frame.copy()

            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out = cv2.resize(out, (320, 240)).astype("float32")
            pred = model.predict(np.expand_dims(out, axis=0))[0][0]

            # if the buffer is not full yet, show the current frames prediction alone
            if len(buffer)<buffer_size:
                buffer.append(pred)
                state = setStateNormal(pred)
            # else show the rolling avg of the previous buffer_size frame's prediction 
            else:
                buffer.pop(0)
                buffer.append(pred)
                state = setStateNormal(sum(buffer)/buffer_size)
            prev_pred = state

            frame = cv2.putText(frame, state, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cur_color, 2)

            frame = cv2.resize(frame, outSize)
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                print(f"Prediction at last frame:\n")
                plt.imshow(frame)
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
