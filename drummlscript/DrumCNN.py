# -*- coding: utf-8 -*-
#Martin Patricio Oña Jativa

import os
from os import walk
import numpy as np
from . import Drum as drum

from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Conv2D, MaxPooling2D

def CNN_custom(midiList,division_tiempo_compas,entrenar_por,path_exit_file,duracion_pista,numero_de_canciones_a_generar,numero_epocas):
    ###############################################################################

    #ruta de las canciones a entrenar
    midiList=midiList

    # se puede dividir en 4, 8 y 16
    division_tiempo_compas=division_tiempo_compas

    #se puede entrenar por tiempo o compas
    entrenar_por=entrenar_por

    #modificar la carpeta siempre de escoger neuvas opciones
    #path_exit_file="../out_datasets/CNN/out_reggae/out_datasets_reggae_4_quarter/generate_song_"
    path_exit_file=path_exit_file

    #si se entrena por compaces un minuto equivale a 30 filas y si es por tiempos 1 min equivale a 120 filas
    duracion_pista=duracion_pista
    #numero de canciones para generar
    numero_de_canciones_a_generar=numero_de_canciones_a_generar

    #numero de epocas a entrenar
    numero_epocas=numero_epocas
    ###############################################################################

    midi_file_X=[]
    midi_file_Y=[]


    for midifile in midiList:
        try: 

            active,jump,steps_per_bar,steps_per_quarter=drum.midiToMatrix(midifile,steps_per_quarter=division_tiempo_compas)
                
            if entrenar_por=='tiempo':
                entrenamiento=steps_per_quarter
            elif entrenar_por=='compas': 
                entrenamiento=steps_per_bar
            
            X_train,Y_train=drum.X_Y_train(active,entrenamiento)
            midi_file_X.append(X_train)
            Y_train=drum.convert2dTo3d(Y_train)
            midi_file_Y.append(Y_train)

        except Exception as e:  
            print(f"Ha ocurrido un error en {midifile}:", type(e).__name__)
            print()


                
    X_train=drum.X_Y_trainConcatenate(midi_file_X)           
    Y_train=drum.X_Y_trainConcatenate(midi_file_Y) 


    arr4d_x_train = np.expand_dims(X_train, 1)
    __,_,filas_tiempo,pitch=arr4d_x_train.shape
    print('n_inputs,channels,steps,pitch:',__,_,filas_tiempo,pitch)

    ############################################################################CNN 
    model = Sequential()

    model.add(Conv2D(5, (3, 3),data_format='channels_first', input_shape=(_,filas_tiempo,pitch)))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense((filas_tiempo*pitch)*2, activation='relu'))
    model.add(Dense(filas_tiempo*pitch, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.summary()
    model.fit(arr4d_x_train, Y_train,epochs=numero_epocas)

    ###############################################################################

    count=0
    while count<=numero_de_canciones_a_generar:
        
        np.random.seed(count)
        print("GENERATING NEW MIDI FILE: ",count)
        exit_midi=path_exit_file+"/"+str(count)+".mid"
        first_temp= np.random.choice([0, 1], size=drum.PITCH_LIMIT*entrenamiento)
        generate_track=[]
        generate_track.append(first_temp)
        new_first_temp=drum.convert3dTo2d(generate_track,entrenamiento)
        new_first_temp= np.expand_dims(new_first_temp, 1)
        new_first_temp=[new_first_temp]
    
        for i in range(duracion_pista):
            
            umbral=model.predict(new_first_temp[-1])
            bridge=np.where(umbral>= np.mean(umbral), 1, 0)
            new=drum.convert3dTo2d(bridge,entrenamiento)
            new=np.array([new])
            new_first_temp.append(new)
            
        new_first_temp=np.array(new_first_temp)
        result = new_first_temp[:,0,0]
        new_final=drum.convert3dToActive(result)
        drum.matrixToMidi(new_final,jump,exit_midi)
        count+=1 



























