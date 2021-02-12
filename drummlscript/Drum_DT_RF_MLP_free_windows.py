# -*- coding: utf-8 -*-
#Martin Patricio Oña Jativa

import os
from os import walk
import numpy as np
from . import Drum as drum
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def DT_RF_MLP_free_windows(midiList,division_tiempo_compas,size_win,duracion_pista,path_exit_file,numero_de_canciones_a_generar,modelo,numero_epocas=None):
    ###############################################################################
    #ruta de las canciones a entrenar
    midiList=midiList
    # se puede dividir en 4, 8 y 16
    division_tiempo_compas=division_tiempo_compas
    #numero de tiempos para entrenar a los modelos 
    size_win=size_win
    #pista duracon
    duracion_pista=duracion_pista
    #salida de la pista
    path_exit_file=path_exit_file
    #numero de canciones a generar
    numero_de_canciones_a_generar=numero_de_canciones_a_generar
    #modelo a escoger
    modelo=modelo
    #epocas a entrenar
    numero_epocas=numero_epocas
    ###############################################################################


    full_train=[]

    for midifile in midiList:
        try: 

            active,jump,__,___=drum.midiToMatrix(midifile,steps_per_quarter=division_tiempo_compas)
            train=drum.activateTrain(active,division_tiempo_compas)
            [full_train.append(train[i]) for i in range(len(train))] 

        except Exception as e:  
            print(f"Ha ocurrido un error en {midifile}:", type(e).__name__)
            print()


    x_train,y_train=drum.freeWin(full_train,size_win)
    print(x_train.shape)
    print('x_train_reshape_foward')
    x_train=drum.x_train_reshape_foward(x_train)
    print(x_train.shape)
    print('x_train_reshape_deep_foward')
    x_train=drum.x_train_reshape_deep_foward(x_train)
    print(x_train.shape)
    print('y_train')
    y_train=drum.convert2dTo3d(y_train)
    print(y_train.shape)

    if modelo=='DecisionTree':
        clf = tree.DecisionTreeClassifier(criterion='entropy')
    elif modelo=='RandomForest':
        clf = RandomForestClassifier(criterion='entropy')
    elif modelo=='MLPClassifier':
        hidden_layers_x,hidden_layers_y=x_train.shape
        #lbfgs
        clf = MLPClassifier(solver='adam',hidden_layer_sizes=hidden_layers_y-300,activation='logistic',max_iter=numero_epocas) 
        print("Epocas de entrenamiento:",numero_epocas)   

    print()
    print(modelo+" FITING...")
    clf = clf.fit(x_train, y_train)
    print("FITING... DONE :) ")
    print()

    count=0
    while count<=numero_de_canciones_a_generar:
        
        np.random.seed(count)
        print(modelo+" GENERATING NEW MIDI FILE: ",count)
        exit_midi=path_exit_file+"/"+str(count)+".mid"
        
        first_temp= np.random.choice([0, 1], size=[1,size_win,division_tiempo_compas,drum.PITCH_LIMIT])
        
        banco_x_first=first_temp
        
        new_drum_track=[]
        _,n_tiempos_first,__,___=banco_x_first.shape
        [new_drum_track.append(banco_x_first[-1][i]) for i in range(n_tiempos_first)]
        
        for song in range(duracion_pista):
        
            banco_x=drum.x_train_reshape_foward(banco_x_first)
            banco_x=drum.x_train_reshape_deep_foward(banco_x)
            
            ##################################
            banco_y=clf.predict([banco_x[-1]])
            ##################################
            
            banco_x=drum.x_train_reshape_deep_back(banco_x,size_win,division_tiempo_compas)
            banco_x=drum.x_train_reshape_back(banco_x,division_tiempo_compas)
            
            banco_y=drum.convert3dTo2d(banco_y,division_tiempo_compas)
            
            new_drum_track.append(banco_y[-1])
            
            new_drum_track_temp=[]
            _,n_tiempos,__,___=banco_x.shape
            [new_drum_track_temp.append(banco_x[-1][i]) for i in range(n_tiempos)]
            new_drum_track_temp.append(banco_y[-1])
            banco_x_first=[new_drum_track_temp[-size_win:]]
            
            
        new_drum_track=np.array(new_drum_track)
        new_file=drum.convert3dToActive(new_drum_track)
        drum.matrixToMidi(new_file,jump,exit_midi)
        count+=1
        



