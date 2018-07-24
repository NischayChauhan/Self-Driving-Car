
# coding: utf-8

# In[ ]:


import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import cv2
from keras.models import load_model
import utils


# In[ ]:


#GETTING DATA FROM THE SIMULATOR VIA A SERVER
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = image[60:-25,:]
        image = cv2.resize(image,(200,66))
        image = np.array([image])  

        steering_angle = float(model.predict(image, batch_size=1))
        
        ## WE WILL INCREASE THE THROTTLE WHICH WILL INCREASE THE SPEED 
        ## HERE YOU CAN PLAY WITH VARIABLES MORE 
        throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

        print steering_angle, throttle, speed
        send_control(steering_angle, throttle)
    else:
        sio.emit('manual', data={}, skip_sid=True)


# In[ ]:


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


# In[ ]:


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={'steering_angle': steering_angle.__str__(),'throttle': throttle.__str__()},
        skip_sid=True)


# In[ ]:


## AND NOW SIMPLY LOAD THE MODEL WHICH YOU HAVE SAVED FROM RUNNING THE TRAIN.IPYNB
# ENJOY THE SIMULATION
model = load_model('./model.h5')
app = socketio.Middleware(sio, app)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

