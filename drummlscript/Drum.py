# -*- coding: utf-8 -*-
#Martin Patricio Oña Jativa

"""Funciones para el procesamiento de las pistas de bateria en formato midi"""

import magenta.music as mm
import matplotlib.pyplot as plt
import numpy as np
import collections
import pandas as pd
from magenta.music import sequences_lib
from . import music_pb2


PITCH_LIMIT=103

def midiToMatrix(midi_file,steps_per_quarter): 
    """
        Args:
            midi_file: Ruta donde se encuentra el archivo midi.
            steps_per_quarter: Division del primer tiempo en 4 semicorcheas, 8 en fusas y 16 en semifusas.

        Returns:
            aa: Matriz binaria, jump: Duracion en tiempo de cada fila, steps_per_bar: Numero de filas en el primer tiempo, steps_per_quarter: Numero de filas del primer compas.
    """
    print(midi_file)
    
    sequence = mm.midi_file_to_sequence_proto(midi_file)
    quantized_sequence = sequences_lib.quantize_note_sequence(sequence, steps_per_quarter=steps_per_quarter)
    drum_song = mm.DrumTrack()
    drum_song.from_quantized_sequence(quantized_sequence, gap_bars=50)
    
    steps_per_bar=drum_song.steps_per_bar
    steps_per_quarter=drum_song.steps_per_quarter
    
    print('steps_per_bar: ',drum_song.steps_per_bar)
    print('steps_per_quarter: ',drum_song.steps_per_quarter)
    
    drum_song=drum_song.to_sequence()
    
    print("total_time",drum_song.total_time)


    count=0
    fila=0
    jump=drum_song.notes[0].end_time - drum_song.notes[0].start_time
    
    print("step_duration",jump)
    
    active = collections.defaultdict(list)
    while count+jump <= drum_song.total_time:
        active['fila'].append(fila)
        active['start_time'].append(count)
        active['end_time'].append(count+jump)
        
        count+=jump
        fila+=1 
    df_1= pd.DataFrame(active)
    #print(df_1)
    
    aa = np.zeros((fila,PITCH_LIMIT))
    print()
    print("WaitMidiToMatrix...........:(")
    for note in drum_song.notes:
        for indice_fila, fila in df_1.iterrows():
            if fila.start_time==note.start_time and note.end_time==fila.end_time :
                aa[int(fila.fila),note.pitch] = 1
                
    print("Done.......................:)")
    print()
    return aa,jump,steps_per_bar,steps_per_quarter
 
    
def matrixToMidi(into_active,jump,exit_midi): 
    """
        Args:
            into_active: Matriz binaria.
            jump: Duracion en tiempo de cada fila.
            exit_midi: Ruta donde se guardaran los nuevos archivos midi.

        Returns:
            Guarda un nuevo archivo midi.
    """
    drums = music_pb2.NoteSequence()
    active = collections.defaultdict(list)
    
    count=0
    x,y=into_active.shape
    
    for i in range(x):
        active['fila'].append(i)
        active['start_time'].append(count)
        active['end_time'].append(count+jump)
        count+=jump
       
    df_1= pd.DataFrame(active)
 
    for i in range(x):
        for j in range(y):
            if into_active[i,j]==1:
                #print(i,j)
                for indice_fila, fila in df_1.iterrows():
                    if indice_fila==i:
                        #print(df_1['start_time'][indice_fila],df_1['end_time'][indice_fila])
                        drums.notes.add(pitch=j, start_time=df_1['start_time'][indice_fila], end_time=df_1['end_time'][indice_fila], is_drum=True, instrument=10, velocity=80)
    
    try:
        drums.total_time = df_1['end_time'][indice_fila]
    except:
        print('La nueva matriz generada por el modelo de aprendizaje automático tiene todos los valores nulos (0)')
        
    drums.tempos.add(qpm=120)
    #print(drums)
    mm.sequence_proto_to_midi_file(drums,exit_midi)

###############################################################################
def graphBinaryMatrix(active,jump,steps_per_bar,steps_per_quarter):
    """
        Args:
            active: Matriz binaria.
            jump: Duracion en tiempo de cada fila.
            steps_per_bar: Numero de filas del primer compas.
            steps_per_quarter: Numero de filas en el primer tiempo.

        Returns:
            Grafica una matriz bianria (partitura).
    """
    
    plt.matshow(active)
    plt.title('Pista de batería cuantizada - matriz binaria')
    plt.xlabel('Notas (Partes de la bateria)')
    plt.ylabel('Tiempo')
    
    x,y=active.shape
   
    count_bar=0
    while count_bar <=x:
        plt.plot([0, PITCH_LIMIT], [ count_bar, count_bar],'c',alpha=1, linewidth=jump)
        count_bar+=steps_per_bar
    count_quarter=0
    while count_quarter <=x:
        
        plt.plot([0, PITCH_LIMIT], [ count_quarter, count_quarter],'w',alpha=1, linewidth=jump)
        count_quarter+=steps_per_quarter
        
        
    plt.xlim(0, PITCH_LIMIT)   
    #plt.ylim(0, 16) 
    plt.show() 


def X_Y_train(active,quarter_or_bar):
    """
        Args:
            active: Matriz binaria.
            quarter_or_bar: Division del conjunto de entrenamiento por primer tiempo o primer compas. 
        Returns:
            X: Valores de entrenamiento (X_train).
            Y: Valores de entrenamiento (Y_train).
    """
    
    x,y=active.shape
    count=0
    count_x_y=0
    X=[]
    Y=[]
    while count <=x:
        if count_x_y %2==0:
            X.append(active[count:count+quarter_or_bar])
            count+=quarter_or_bar
            count_x_y+=1
        else:
            Y.append(active[count:count+quarter_or_bar])
            count+=quarter_or_bar
            count_x_y+=1
            
    zero=np.zeros(PITCH_LIMIT)
    
    long_0= len(X[0])
    long_ny=len(Y[len(Y)-1])
    long_nx=len(X[len(X)-1])

    if long_0!=long_ny:
        for i in range(long_0-long_ny):
            Y[len(Y)-1]=np.vstack([Y[len(Y)-1], zero])
    
    if long_0!=long_nx:
        for i in range(long_0-long_nx):
            X[len(X)-1]=np.vstack([X[len(X)-1], zero])
        
    if len(X)!=len(Y):
        complete=[]
        for i in range(long_0):
            complete.append(zero)
        complete=np.array(complete)
        Y.append(complete)

    X=np.array(X)
    Y=np.array(Y)
    X=np.int_(X)
    Y=np.int_(Y)
    
    return X,Y

def convert2dTo3d(X):
    """
        Args:
            X: Conjunto de entrenamiento.
        Returns:
            matrix_2d: Matriz con una nueva estructura.
    """
    nsamples, nx, ny = X.shape
    matrix_2d= X.reshape((nsamples,nx*ny))
    return matrix_2d


def convert3dTo2d(X,quarter_or_bar):
    """
        Args:
            X: Conjunto de entrenamiento.
            quarter_or_bar: Reorganiza a la mtriz en la division de entrenamiento primer tiempo o primer compas.   
        Returns:
            matrix_3d: Matriz con una nueva estructura.
    """
    X=np.array(X)
    x,y=X.shape
    matrix_3d=X.reshape(x,quarter_or_bar,PITCH_LIMIT)
    return matrix_3d
    
def convert3dToActive(X):
    """
        Args:
            X: Matriz generada por los modelos de aprendisaje automatico.
        Returns:
           activate: Matriz con el formato (dimensiones) para generar la nueva pista de bateria.  
    """
    active=[]
    x,y,z=X.shape
    
    for i in range(x):
        for j in range(y):
            active.append(X[i][j])      
    active=np.array(active)
    return active

def X_Y_trainConcatenate(midi_file):
    """
        Args:
            midi_file:      .
        Returns:
            X_base:         .  
    """
    midi_file=np.array(midi_file)
    for indice,valor in enumerate(midi_file):
        if indice ==0:
            X1=valor
        if indice ==1:
            X2=valor
            X_base=np.concatenate((X1, X2), axis=0)
            
        if indice>1:
            X_base=np.concatenate((X_base,valor),axis=0)
        
    return X_base
###############################################################################
# FOR FREE SIZE WINDOWS
def freeWin(activate,size_win):
  x_train=[]
  y_train=[]
  for i in range(len(activate)-size_win):
    x_train.append(activate[i:i+size_win])
    y_train.append(activate[i+size_win])
  
  x_train,y_train=np.array(x_train),np.array(y_train)
  return x_train,y_train


def activateTrain(activate,name_note):
  count=0
  train=[]
  while count <=len(activate):
    train.append(activate[count:count+name_note])
    count+=name_note
    
  zero=np.zeros(PITCH_LIMIT)
  
  if len(train[-1])<name_note:
    while len(train[-1])<name_note:
      train[-1]=np.vstack([train[-1],zero])
  
  train=np.array(train)
  return train

def x_train_reshape_foward(x):
  new=[]
  x=np.array(x)
  n_samples,_,__,pitch=x.shape
  for i in range(n_samples):
    new.append(convert2dTo3d(x[i]))
  new=np.array(new)
  return new 

def x_train_reshape_back(x,name_note):
  n_samples,_,__=x.shape
  new=[]
  for i in range(n_samples):
      new.append(convert3dTo2d(x[i],name_note))
  new=np.array(new)
  return new
 
def x_train_reshape_deep_foward(x):
  x=np.array(x)
  new=(convert2dTo3d(x))
  new=np.array(new)
  return new

def x_train_reshape_deep_back(matrix,size_win,nate_note):
  n_samples,_=matrix.shape
  matrix_3d=matrix.reshape(n_samples,size_win,nate_note*PITCH_LIMIT)
  matrix_3d=np.array(matrix_3d)
  return matrix_3d
