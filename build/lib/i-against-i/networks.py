from iai.imports import *

def GRU_TUNED84(x,y):
    '''
    Same as GRU_VANILLA but with dropout AFTER each dense layer.
    '''
    haps,pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = Input(shape=(numSNPs,numSamps))
    model = layers.Bidirectional(layers.CuDNNGRU(84,return_sequences=False))(genotype_inputs)
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)

    #----------------------------------------------------

    position_inputs = Input(shape=(numPos,))
    m2 = Dense(256)(position_inputs)

    #----------------------------------------------------


    model =  layers.concatenate([model,m2])
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1)(model)

    #----------------------------------------------------

    model = Model(inputs=[genotype_inputs,position_inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def GRU_POOLED(x,y):

    sites=x.shape[1]
    features=x.shape[2]

    genotype_inputs = Input(shape=(sites,features))
    model = layers.Bidirectional(layers.CuDNNGRU(84,return_sequences=False))(genotype_inputs)
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1)(model)

    model = Model(inputs=[genotype_inputs], outputs=[output])
    model.compile(optimizer='Adam', loss='mse')
    model.summary()

    return model


def HOTSPOT_CLASSIFY(x,y):

    haps,pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = Input(shape=(numSNPs,numSamps))
    model = layers.Bidirectional(layers.CuDNNGRU(84,return_sequences=False))(genotype_inputs)
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)

    #----------------------------------------------------

    position_inputs = Input(shape=(numPos,))
    m2 = Dense(256)(position_inputs)

    #----------------------------------------------------


    model =  layers.concatenate([model,m2])
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1,activation='sigmoid')(model)

    #----------------------------------------------------

    model = Model(inputs=[genotype_inputs,position_inputs], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    return model
