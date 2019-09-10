#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense , Flatten , Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image


# In[ ]:


datagen = image.ImageDataGenerator(rescale = 1. / 255,
                                  validation_split = 0.2,
                                  rotation_range = 20,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,
                                  fill_mode = 'nearest')


# In[ ]:


pathname = './fruits-360/Training/'
train_generator = datagen.flow_from_directory(pathname,
                                             target_size = (100,100),
                                             batch_size = 45852,
                                             subset = 'training',
                                             class_mode = 'categorical')


# In[ ]:


pathtotest = './fruits-360/Test/'
test_generator = datagen.flow_from_directory(pathtotest,
                                             target_size = (100,100),
                                             batch_size = 5000,
                                             subset = 'training',
                                             class_mode = 'categorical')


# In[ ]:


x_train, y_train = next(train_generator)


# In[ ]:


x_test, y_test = next(test_generator)


# In[ ]:


#pathname = './fruits-360/Training/'
#x = []
#y = []
#num_classes = 0
#for dirname, aashish, filenames in os.walk(pathname) :
#	#print(dirname)
#	#print(type(dirname))
#	num_classes += 1
#	label = dirname.split("/")[-1]
#	print(label)
#	for filename in filenames :
##       #img = image.load_img(pathtofile)
#        #x.append(img)
#		y.append(label)	
		#print(type(img))
	#print(filenames)
#	print("-------------------------------------------")
#	if num_classes == 5 :
#		break


# In[ ]:


#pathname = './fruits-360/Training/'
#x = []
#y = []
#num_classes = 0
#labels = []

#for dirname, aashish, filenames in os.walk(pathname) :
#    label = dirname.split("/")[-1]
#    
#    print(label)
#    labels.append(label)
#   for filename in filenames :
#        pathtofile = os.path.join(dirname, filename)
#        img = image.load_img(pathtofile, target_size = (100,100))
#        p = image.img_to_array(img)
#        x.append(p)
#        y.append(num_classes)
#    num_classes += 1
#    if num_classes == 5 :
#        break


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


_, num_classes = y_train.shape
num_classes


# In[ ]:


#labels = list(set(y))
#num_classes = len(labels)
#labels = labels[1:]
#labels


# In[ ]:


#x_new = []
#for ele in x :
#    x_new.append(ele / 255)
    #print(ele)
#print(x_new)


# In[ ]:


#dictOfLabels = { labels[i] : i for i in range(0, len(labels) ) }
#print(dictOfLabels)
#tf.feature_column.categorical_column_with_vocabulary_list('labels', labels)
#print(labels)

#out_y = tf.keras.utils.to_categorical(y, num_classes)
#y_new = []
#for ele in y :
#    y_new.append(dictOfLabels[ele])
#print(y[:10])
#print(y_new[:10])


# In[ ]:


#now spliting tha data into train and test
#x_train, x_test, y_train, y_test = train_test_split(x_new,y, test_size = 0.2, shuffle = True)


# In[ ]:


print(len(x_train))
print(len(x_test))
#y_new


# In[ ]:


model = tf.keras.models.Sequential()
model.add(Conv2D(16, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2), padding = 'same'))

model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2), padding = 'same'))

model.add(Conv2D(48, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2), padding = 'same'))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2), padding = 'same'))

model.add(Conv2D(num_classes, kernel_size = (3, 3), activation = 'relu'))
model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=5,
          validation_split = 0.2)
#data preperation
#model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




