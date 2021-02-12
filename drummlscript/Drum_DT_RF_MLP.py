# -*- coding: utf-8 -*-
#Martin Patricio Oña Jativa

"""Generacion de pistas de bateria"""

from sklearn import tree
#import graphviz 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import os
from os import walk
import numpy as np
from . import Drum as drum

def DT_RF_MLP(midiList,division_tiempo_compas,entrenar_por,path_exit_file,duracion_pista,modelo,numero_de_canciones_a_generar,numero_epocas=None):

    midiList=midiList
    #path_open_file=path_open_file
    division_tiempo_compas=division_tiempo_compas
    entrenar_por=entrenar_por
    path_exit_file=path_exit_file
    duracion_pista=duracion_pista
    modelo=modelo
    numero_de_canciones_a_generar=numero_de_canciones_a_generar
    numero_epocas=numero_epocas

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
            X_train=drum.convert2dTo3d(X_train)
            Y_train=drum.convert2dTo3d(Y_train)
            midi_file_X.append(X_train)
            midi_file_Y.append(Y_train) 
        except Exception as e:  
            print(f"Ha ocurrido un error en {midifile}:", type(e).__name__)
            print()

    X_train=drum.X_Y_trainConcatenate(midi_file_X)           
    Y_train=drum.X_Y_trainConcatenate(midi_file_Y) 


    if modelo=='DecisionTree':
        clf = tree.DecisionTreeClassifier(criterion='entropy')
    elif modelo=='RandomForest':
        clf = RandomForestClassifier(criterion='entropy')
    elif modelo=='MLPClassifier':
        hidden_layers_x,hidden_layers_y=X_train.shape
        #lbfgs
        clf = MLPClassifier(solver='adam',hidden_layer_sizes=hidden_layers_y-300,activation='logistic',max_iter=numero_epocas)    


    print(modelo+" FITING...")
    clf = clf.fit(X_train, Y_train)
    print("FITING... DONE :) ")
    print()

    count=0
    while count<=numero_de_canciones_a_generar:
        np.random.seed(count)
        print(modelo+" GENERATING NEW MIDI FILE: ",count)
        
        exit_midi=path_exit_file+"/"+str(count)+".mid"
        print(exit_midi)
        first_temp= np.random.choice([0, 1], size=drum.PITCH_LIMIT*entrenamiento)
        generate_track=[]
        generate_track.append(first_temp)
        
        for i in range(duracion_pista):
            bridge=clf.predict([generate_track[-1]])
            generate_track.append(bridge[0])
        
        
        new=drum.convert3dTo2d(generate_track,entrenamiento)
        new=drum.convert3dToActive(new)
        drum.matrixToMidi(new,jump,exit_midi)
        count+=1        
        

