# DS19 Workshop | Building a Car Damage Classification model using Google Colab

The below markdown file consists of commands and code snippets that will help you complete the lab - Building a Car Damage Classification model using Google Colab

## Code Snippets

#### Cloning the Github repository
```
!git clone https://github.com/ammu11/DS19-DamageCarClassification
```
 #### Import required libraries
```
import keras
import os
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras import regularizers
from keras.layers import Dense,GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import os.path as osp
import argparse
import tensorflow as tf
from keras.models import load_model
```
#### Train and validation folder paths

```
train_dir = "/content/DS19-DamageCarClassification/training"
test_dir = "/content/DS19-DamageCarClassification/testing"
```
#### Considering the model 
```
base_model=MobileNet(weights='imagenet',include_top=False)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x=Dense(512,activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
preds=Dense(3,activation='softmax', kernel_regularizer=regularizers.l2(0.001))(x)
```

#### Specify the model with respective input and output parameters
```
model=Model(inputs=base_model.input,outputs=preds)
```
#### Fine tuning the model
```
for layer in model.layers[:20]:
layer.trainable=False
for layer in model.layers[20:]:
layer.trainable=True
```
#### Data Augmentation
```
img_width, img_height = 224, 224
epochs = 200
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
test_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode='categorical')
```

#### Creation of checkpoints
```
import os
from keras.callbacks import ModelCheckpoint
savepath = os.path.join( ""+ 'e1-{epoch:03d}-vl-{val_loss:.3f}-va-{val_acc:.3f}.h5' )
checkpointer = ModelCheckpoint(filepath=savepath,monitor='val_acc', mode='max', verbose=0, save_best_only=True)
```

#### Train the model 
```
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit_generator(
train_generator,
steps_per_epoch=train_generator.samples // batch_size,
epochs=epochs,
validation_data=validation_generator,
validation_steps=validation_generator.samples // batch_size,
callbacks=[checkpointer])
```
#### Plot the model accuracy and loss
```
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
#### Save the model with filename
```
model.save_weights("model.h5")
```
#### Access the model
```
saved_model="/content/DS19-DamageCarClassification/path_to_the_last_checkpoint_file.h5"
# (or)
#Access the last checkpoint file as final model
saved_model="/content/checkpoint-025.h5"
```
#### Test the model
```
def  predict(model, img):
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	return preds[0]

labels = ("Bumper_Damage","Door_Damage","Glass_Damage")
model = load_model(saved_model)
img = image.load_img('/content/DS19-DamageCarClassification/testing/bumper/0048.JPEG', target_size=(224,224))
preds = predict(model, img)
j=max(preds)
result = np.where(preds == j)
index_val = result[0][0]
prediction = labels[index_val]
print("Result:",prediction)
```



