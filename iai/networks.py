'''
Authors: Jeff Adrion
'''

from iai.imports import *


def iaiCNN_categorical_crossentropy(inputShape,y):

    haps,pos = inputShape

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    img_1_inputs = Input(shape=(numSNPs,numSamps))

    h = layers.Conv1D(1250, kernel_size=2, activation='relu', name='conv1_1')(img_1_inputs)
    h = layers.Conv1D(512, kernel_size=2, dilation_rate=1, activation='relu')(h)
    h = layers.AveragePooling1D(pool_size=2)(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Conv1D(512, kernel_size=2, activation='relu')(h)
    h = layers.AveragePooling1D(pool_size=2)(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Flatten()(h)

    loc_input = Input(shape=(numPos,))
    m2 = layers.Dense(64,name="m2_dense1")(loc_input)
    m2 = layers.Dropout(0.1)(m2)

    h =  layers.concatenate([h,m2])
    h = layers.Dense(128,activation='relu')(h)
    h = Dropout(0.2)(h)
    output = layers.Dense(2,kernel_initializer='normal',name="softmax",activation='softmax')(h)

    model = Model(inputs=[img_1_inputs,loc_input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def iaiCNN_binary_crossentropy(inputShape,y):

    haps,pos = inputShape

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    img_1_inputs = Input(shape=(numSNPs,numSamps))

    h = layers.Conv1D(1250, kernel_size=2, activation='relu', name='conv1_1')(img_1_inputs)
    h = layers.Conv1D(512, kernel_size=2, dilation_rate=1, activation='relu')(h)
    h = layers.AveragePooling1D(pool_size=2)(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Conv1D(512, kernel_size=2, activation='relu')(h)
    h = layers.AveragePooling1D(pool_size=2)(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Flatten()(h)

    loc_input = Input(shape=(numPos,))
    m2 = layers.Dense(64,name="m2_dense1")(loc_input)
    m2 = layers.Dropout(0.1)(m2)

    h =  layers.concatenate([h,m2])
    h = layers.Dense(128,activation='relu')(h)
    h = Dropout(0.2)(h)
    output = layers.Dense(1,kernel_initializer='normal',name="out_dense",activation='sigmoid')(h)

    model = Model(inputs=[img_1_inputs,loc_input], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    return model


def iaiCNN_adv(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):

    model = Sequential()

    input_shape = (img_rows, img_cols)

    layers = [Conv1D(1250, kernel_size=2, activation='relu', name='conv1_1', input_shape=input_shape),
            Conv1D(512, kernel_size=2, dilation_rate=1, activation='relu'),
            AveragePooling1D(pool_size=2),
            Dropout(0.25),
            Conv1D(512, kernel_size=2, activation='relu'),
            AveragePooling1D(pool_size=2),
            Dropout(0.25),
            Flatten(),
            Dense(128,activation='relu'),
            Dropout(0.2),
            Dense(nb_classes)]

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model

