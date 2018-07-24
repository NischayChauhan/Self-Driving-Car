
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential, save_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten


# In[ ]:


# THIS WILL CUT YOUR IMAGE SUCH THAT ONLY THAT PART WHICH CONTAIN ROAD IN IT REMAINS.
def processImage(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image[60:-25,:]
    return image


# WE ARE TAKING ONLY CENTER IMAGE AND STERRING ANGLE AND WILL COMPUTE THROTTLE ACCORDINGLY BUT YOU CAN EXPERIMENT WITH IT
def getData(ds):
    ds = np.array(ds)
    centerImageLoaction = ds[:,0]
    SterringAngle = ds[:,3]
    newImage = []
    for i in centerImageLoaction:
        newImage.append(processImage(i))
    newImage = np.array(newImage)
    sterringAngle = ds[:,3]
    return newImage, sterringAngle


# In[ ]:


# THIS LOG WILL BE THERE ONCE YOU RUN THE SIMULATOR IN TRAINING MODE (and PRESS R TO RECORD)
ds = pd.read_csv('./driving_log.csv')
X,Y = getData(ds)


# In[ ]:


print (X.shape,Y.shape)


# In[ ]:


plt.imshow(X[40])
plt.show()
print( Y[40])


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

print( X_train.shape)
print (X_test.shape)

print( Y_train.shape)
print (Y_test.shape)


# In[ ]:


X_train = X_train/255.0
X_test = X_test/255.0


# In[ ]:


model = Sequential()
model.add(Conv2D(24, 5, 5, activation='relu',input_shape = (75, 320, 3), subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()


# In[ ]:


checkpoint = ModelCheckpoint('model-{val_loss:.4f}.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit( X_train, Y_train, batch_size=30, nb_epoch=50, validation_data=(X_test, Y_test),callbacks=[checkpoint], shuffle=True )

