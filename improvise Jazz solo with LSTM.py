# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:53:56 2020

@author: z
"""

import IPython


from keras.utils import to_categorical
from keras.optimizers import Adam


X, Y, n_values, indices_values = load_music_utils()
n_a = 64

def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    
    model -- a keras model with the 
    """

    X = Input(shape=(Tx, n_values)) 

    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    outputs = [] 

    for t in range(Tx):
        
        x = Lambda(lambda x: X[:,t,:])(X)
        x = reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)

    model = Model(inputs=[X, a0, c0], outputs=outputs)
    return model

i = 0
while i<1:  
    model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    model.fit([X, a0, c0], list(Y), epochs=100)
    i +=1

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    inference_model -- Keras model instance
    """
    
    x0 = Input(shape=(1, n_values))
    
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []

    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        out = densor(a)

        outputs.append(out)

        x = Lambda(one_hot)(out)
        
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    return inference_model

j = 0
while j < 1:    
    inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)
    x_initializer = np.zeros((1, 1, 78))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))
    
def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, axis=-1)
    results = to_categorical(indices, num_classes=78)
    
    return results, indices

results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
out_stream = generate_music(inference_model)
IPython.display.Audio('./data/30s_trained_model.mp3')



