# -*- coding: utf-8 -*-
#Martin Patricio Oña Jativa

"""Informacion de los archivos MIDI"""

from mido import MidiFile,MidiTrack
import collections
import pandas as pd

def midiInfo(ruta,on_off):  
    """
        Args:
            ruta: Ruta donde se encuentra el archivo midi.
            on_off: Mostrar o no los mensajes.

        Returns:
            time_compas: Duracion de cada compas, total_compas: Numero total de compaces.
    """
    mid = MidiFile(ruta)
    print(ruta)
    #print(f'ticks_per_beat: {mid.ticks_per_beat}')
    #print(f'length: {mid.length}')
    for i, track in enumerate(mid.tracks):
            #print('Track {}: {}'.format(i, track.name))
            
            for i,msg in enumerate(track):
                    if msg.type=='set_tempo':
                        a=msg.tempo
                    elif msg.type=='time_signature':
                        b=msg.numerator
                    if on_off==True:
                        print(i,msg)
                    
    time_compas=((a/1000000)*b)
    #total_compas=math.ceil(mid.length/time_compas)
    total_compas=mid.length/time_compas
    print(f'La duracion de cada compas es (s): {time_compas}')
    print(f'La pista de bateria tiene un total de compaces de: {total_compas}')
    return time_compas,total_compas
                                
def midiChangeChannel(ruta,exit_root):
    """
        Args:
            ruta: Ruta donde se encuentra el archivo midi.
            exit_root: Ruta donde se guardaran los nuevos archivos midi.

        Returns:
            Guarda un nuevo archivo midi.
    """
    mid = MidiFile(ruta)
    newmid = MidiFile()
    newtrack = MidiTrack()
    newmid.tracks.append(newtrack)
    
    if mid.ticks_per_beat != newmid.ticks_per_beat:
            newmid.ticks_per_beat=mid.ticks_per_beat

    for i, track in enumerate(mid.tracks):
        pass

    for i,msg in enumerate(track):
        if msg.is_meta==False :
            #print(i,msg)
            msg.channel=9
            
        newtrack.append(msg)

    newmid.save(exit_root)
    
def sequenceToPandasDataframe(sequence):
    """Generates a pandas dataframe from a sequence.
    
        Args:
            sequence: sequence of MIDI.

        Returns:
            DataFrame: DataFrame of MIDI.
    """
    
    pd_dict = collections.defaultdict(list)
    for note in sequence.notes:
        pd_dict['start_time'].append(note.start_time)
        pd_dict['end_time'].append(note.end_time)
        pd_dict['duration'].append(note.end_time - note.start_time)
        pd_dict['pitch'].append(note.pitch)
        pd_dict['velocity'].append(note.velocity)
        pd_dict['instrument'].append(note.instrument)
        #pd_dict['program'].append(note.program)
    return pd.DataFrame(pd_dict)

