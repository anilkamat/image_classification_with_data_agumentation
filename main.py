#%%
from tkinter.tix import COLUMN
import tensorflow as tf
from PIL import Image
import os
import numpy as np

#%%
breeds = ["beagle", "bernese_mountain_dog", "doberman", "labrador_retriever", "siberian_husky"]

args={
    'labels':'inferred',
    'label_mode':'categorical',
    'batch_size':32,
    'image_size':(256,256),
    'seed':1,
    'validation_split':0.2,
    'class_names':breeds
}

# args = {
#     "labels": "inferred",
#     "label_mode": "categorical",
#     "batch_size": 32,
#     "image_size": (256, 256),
#     "seed": 1,
#     "validation_split": .2,
#     "class_names": breeds
# }
PATH = r'C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\GNN\pythonCodes\TF_models\Model1\images'
train = tf.keras.utils.image_dataset_from_directory(
    PATH,
    subset="training",
    **args
)

test = tf.keras.utils.image_dataset_from_directory(
    PATH,
    subset='validation',
    **args
)
# tf.keras.utils.image_dataset_from_directory(PATH,subset='training',**args)
#%% 
first = train.take(1)
first
images,labels = list(first)[0]
first_image = images[0]
first_image = first_image[:32,:32,:]
#%%
#Image.fromarray(first_image.numpy().astype("uint8"))
Image.fromarray(images[0].numpy().astype("uint8"))

#%%
train = train.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
test = test.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

#%%
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#%%
#Model 
Model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16,3,padding='same',activation='relu',input_shape= (256,256,3)),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(len(breeds))
])
#%%
Model.compile(optimizer= 'adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True),metrics= ['accuracy'])

#%%
history = Model.fit(
    train,
    validation_data=test,
    epochs = 5,
    verbose=1
)
#%%
Model.summary()
#%%
import pandas as pd
history_df = pd.DataFrame.from_dict(history.history)
history_df[['accuracy','loss','val_accuracy']].plot()
#%%
def train_model(Network, epochs = 5 ):
    model = Sequential(Network)
    Model.compile(optimizer= 'adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True),metrics= ['accuracy'])
    history = Model.fit(
    train,
    validation_data=test,
    epochs = epochs,
    verbose=1)
    history_df = pd.DataFrame.from_dict(history.history)
    return history_df, model

#%%
Network = [
    layers.Rescaling(1./255),
    layers.Conv2D(16,3,padding='same',activation='relu',input_shape= (256,256,3)),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    layers.Conv2D(32,3,padding='same',activation='relu',input_shape= (256,256,3)),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    layers.Conv2D(64,3,padding='same',activation='relu',input_shape= (256,256,3)),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(len(breeds))
]
#%%
hist , model = train_model(Network, 10)
#%%
hist[['accuracy','loss']].plot()
#%%
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal', seed = 1),
    layers.RandomZoom(0.2, seed = 2),
    layers.RandomRotation(0.2, seed=3)
])
#%%
Full_network = [data_augmentation]+Network

#%%
hist, model = train_model(Full_network, 5)
hist[['accuracy','loss']].plot()

#%%
preds = model.predict(test)
#print(preds)

#%%
preds = np.argmax(preds, axis=1)
preds
#%%
actual_labels = np.concatenate([y for x,y in test], axis=0)
actual_class  = np.argmax(actual_labels, axis=1 )

#%%
import itertools
actual_images = [x.numpy().astype('uint8') for x,y in test]
actual_images = list(itertools.chain.from_iterable(actual_images))
actual_images = [Image.fromarray(x) for x in actual_images]

#%%
df = pd.DataFrame(zip(actual_class,preds,actual_images), columns=['actual_class','preds','actual_images'])
#%%
df['preds'] = df['preds'].apply(lambda x: breeds[x])
df['actual_class'] = df['actual_class'].apply(lambda x: breeds[x])

#%%
df.head()

#%%
import base64
import io

def image_formatter(img):
    with io.BytesIO() as buffer:
        img.save(buffer, 'png')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f'<img src="data:image/jpeg;base64,{img_str}">'

df.head(5).style.format({'image': image_formatter})
