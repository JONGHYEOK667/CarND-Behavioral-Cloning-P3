#!/usr/bin/env python
# coding: utf-8

# ---
# ## Step 0: Load The Data

# In[1]:


import tensorflow as tf
print(tf.__version__)

import keras
print(keras.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory rowth must be set before GPUs have been initialized
    print(e)
    
# config = tf.ConfigProto()
# config.gpu_option.per_process_gpu_memory_fraction = 0.4
# session = tf.Session(config=config)
# session.close()


# In[2]:


import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle
import seaborn as sns


# In[3]:


lines = []
center_img, left_img, right_img, steer_val = [],[],[],[]

path = '../../../Udacity/self-driving-car-game/0.driving_data/driving_log.csv'

with open(path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if float(line[3]) != 0.0:
            center_img.append(line[0])
            left_img.append(line[1])
            right_img.append(line[2])
            steer_val.append(float(line[3]))
            
        else:
            prob = np.random.uniform()
            if prob <= 0.2:
                center_img.append(line[0])
                left_img.append(line[1])
                right_img.append(line[2])
                steer_val.append(float(line[3]))
            
            

f = plt.hist(steer_val, bins = 40, edgecolor='black', linewidth = 1.2)
plt.title('Collected data', fontsize = 10)
plt.xlabel('Steering value (scaled)')
plt.ylabel('counts')
# plt.savefig('output_fig/1.collected_data.jpg')

steer_val = np.array(steer_val)
steer_val = np.around(steer_val,3)


# In[4]:



def BGR2RGB(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_RGB


def extract_array(img_path):
    images = []
    for line in img_path:
        source_path = line
        filename = source_path.split('\\')[-1]
        current_path = '../../../Udacity/self-driving-car-game/0.driving_data/IMG/' + filename
        image = cv2.imread(current_path)
        image = BGR2RGB(image)
        images.append(image)
    return images



def Flip(imgs):
    images = []
    for img in imgs:
        image = cv2.flip(img,1)
        images.append(image)
        
    return images


# In[5]:


## make left / center / right img, steer data

offset = 0.1

images_left = extract_array(left_img)
steers_left = steer_val + offset

images_center = extract_array(center_img)
steers_center = steer_val

images_right = extract_array(right_img)
steers_right = steer_val - offset


# In[6]:


images_left_flip = Flip(images_left)
steers_left_flip = -steers_left


images_center_flip = Flip(images_center)
steers_center_flip = -steers_center


images_right_flip = Flip(images_right)
steers_right_flip = -steers_right


# In[7]:


index = np.random.randint(len(steer_val)+1)

image_left = images_left[index]
image_center = images_center[index]
image_right = images_right[index]

steer_left = steers_left[index]
steer_center = steers_center[index]
steer_right = steers_right[index]


image_left_flip = images_left_flip[index]
image_center_flip = images_center_flip[index]
image_right_flip = images_right_flip[index]

steer_left_flip = steers_left_flip[index]
steer_center_flip = steers_center_flip[index]
steer_right_flip = steers_right_flip[index]



f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(15, 10))
ax1.imshow(image_left)
ax1.set_title('left, '+'steer('+str(np.around(steer_left,3))+')', fontsize=20)
ax2.imshow(image_center)
ax2.set_title('center, '+'steer('+str(np.around(steer_center,3))+')', fontsize=20)
ax3.imshow(image_right)
ax3.set_title('right, '+'steer('+str(np.around(steer_right,3))+')', fontsize=20)
ax4.imshow(image_left_flip)
ax4.set_title('left_flip, '+'steer('+str(np.around(steer_left_flip,3))+')', fontsize=20)
ax5.imshow(image_center_flip)
ax5.set_title('center_flip, '+'steer('+str(np.around(steer_center_flip,3))+')', fontsize=20)
ax6.imshow(image_right_flip)
ax6.set_title('right_flip, '+'steer('+str(np.around(steer_right_flip,3))+')', fontsize=20)

f.tight_layout()
f.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.0)


# In[8]:


# f.savefig('output_fig/2.Augmented_Image(BiasedRecover).jpg')


# In[9]:


images = images_left + images_center + images_right         + images_left_flip + images_center_flip + images_right_flip
images = np.array(images)

steers = np.append(steers_left,steers_center)
steers = np.append(steers,steers_right)
steers = np.append(steers,steers_left_flip)
steers = np.append(steers,steers_center_flip)
steers = np.append(steers,steers_right_flip)

# images =  images_center +  images_center_flip 
# images = np.array(images)

# steers = np.append(steers_center,steers_center_flip)



f = plt.hist(steers, bins = 40, edgecolor='black', linewidth = 1.2)
plt.title('Augmented data', fontsize = 10)
plt.xlabel('Steering value (scaled)')
plt.ylabel('counts')
# plt.savefig('output_fig/3.Augmented_data.jpg')


# In[10]:


index = np.random.choice(steers.shape[0], int(steers.shape[0]/1), replace = False)
x_suffle = images[index]
y_suffle = steers[index]


# In[11]:


x_train = x_suffle[0:int(7*steers.shape[0]/10)]
y_train = y_suffle[0:int(7*steers.shape[0]/10)]

x_val = x_suffle[int(7*steers.shape[0]/10):]
y_val = y_suffle[int(7*steers.shape[0]/10):]


display(x_suffle.shape)
display(y_suffle.shape)

display(x_train.shape)
display(y_train.shape)

display(x_val.shape)
display(y_val.shape)


# In[12]:


def generator(feature, label, batch_size = 32):
    num_sample = feature.shape[0]
    display(num_sample)
    display(range(0, num_sample, batch_size))
    while 1 :
        for offset in range(0, num_sample, batch_size):
            x_train = feature[offset:offset+batch_size]
            y_train = label[offset:offset+batch_size]

            yield (x_train, y_train)     
    


# In[13]:


batch_size = 256
train_generator = generator(x_train, y_train, batch_size = batch_size)
val_generator = generator(x_val, y_val, batch_size = batch_size)


# In[14]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Cropping2D

drop_rate = 0.4

model = keras.models.Sequential([
    keras.layers.Cropping2D(cropping=((70,25),(0,0)),input_shape = (160, 320, 3)),
    keras.layers.Lambda(lambda x : x/255.0 - 0.5),
    keras.layers.Conv2D(filters = 24,kernel_size = (5,5), strides = (2,2),  padding = 'same'),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(filters = 36,kernel_size = (5,5), strides = (2,2),  padding = 'same'),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(filters = 48,kernel_size = (5,5), strides = (2,2),  padding = 'same'),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1),  padding = 'same'),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1),  padding = 'same'),
    keras.layers.Activation('relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation = 'relu'),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(50, activation = 'relu'),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(10, activation = 'relu'),
    keras.layers.Dense(1)
])


model.summary()
model.compile(optimizer = keras.optimizers.Adam(),
             loss = 'mse', metrics = ['mae'])


# In[15]:


from math import ceil

steps_per_epoch = ceil(x_train.shape[0]/batch_size)
validation_steps = ceil(x_val.shape[0]/batch_size)

display(steps_per_epoch)
display(validation_steps)


# In[16]:


history = model.fit_generator(train_generator, 
                              steps_per_epoch = steps_per_epoch,
                              validation_data = val_generator,
                              validation_steps = validation_steps,
                              epochs = 100,
                              callbacks = [keras.callbacks.EarlyStopping(patience=5,monitor='val_loss',mode = 'min',verbose = 1 )],
                              verbose = 1)


# In[17]:


# history = model.fit(x_train, y_train, epochs = 100, batch_size = 256,
#                     callbacks = [keras.callbacks.EarlyStopping(patience=5,monitor='val_loss',mode = 'min',verbose = 1 )],
#                     validation_split = 0.3 , verbose = 1, shuffle = True  )


# In[18]:


history.history
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.plot(history.history['loss'], '-b', label = 'loss')
ax1.plot(history.history['val_loss'], '--b', label = 'val_loss')
ax2.plot(history.history['mae'], '-r', label = 'mae')
ax2.plot(history.history['val_mae'], '--r', label = 'val_mae')
ax1.set_title('loss (mse)', fontsize=20)
ax1.set_xlabel('Epoch')
ax2.set_title('mae', fontsize=20)
ax2.set_xlabel('Epoch')
f.tight_layout()
f.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# f.savefig('output_fig/4.Train_History.jpg')


# In[19]:


model.save('model_temp.h5')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




