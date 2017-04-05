# https://keras.io/getting-started/functional-api-guide/#shared-layers

from keras.models import Model
from keras.layers import Input, Convolution2D, GlobalAveragePooling2D, Dense
import numpy as np

shared_layer = Convolution2D(128, 3, 3)

input1 = Input((32,32,3))
x = shared_layer(input1)
x = GlobalAveragePooling2D()(x)
output1 = Dense(2, activation='softmax')(x)
model1 = Model(input=input1, output=output1)
print(model1.summary())

input2 = Input((64,64,3))
x = shared_layer(input2)
x = GlobalAveragePooling2D()(x)
output2 = Dense(2, activation='softmax')(x)
model2 = Model(input=input2, output=output2)
print(model2.summary())

model1.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model2.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model1.fit(np.zeros((16,32,32,3)), np.zeros((16,2)), batch_size=16, nb_epoch=1, verbose=1)
model2.fit(np.zeros((8,64,64,3)), np.zeros((8,2)), batch_size=8, nb_epoch=1, verbose=1)
