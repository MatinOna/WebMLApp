from django.http import HttpResponseRedirect
from django.shortcuts import render,redirect,HttpResponse
from django.urls import reverse
from drummlscript.Drum_DT_RF_MLP  import DT_RF_MLP
from drummlscript.Drum_DT_RF_MLP_free_windows  import DT_RF_MLP_free_windows
from drummlscript.Drum_LSTM_free_windows import LSTM_free_windows
from drummlscript.Drum_LSTM import LSTM_custom
from drummlscript.DrumCNN_free_windows import CNN_free_windows
from drummlscript.DrumCNN import CNN_custom
from django.core.files.storage import FileSystemStorage
from django.conf import settings

def home(request):
    return render(request,"core/home.html")

def uno(request):
    dic = {}
    midiList=[]
    if request.method == 'POST':
        files=request.FILES.getlist('midi')
        numero_canciones=int(request.POST.get('n_song'))
        modelo=request.POST.get('gender')
        division_tiempo_compas=int(request.POST.get('division'))
        size_win=int(request.POST.get('sizew'))
        long_canciones=int(request.POST.get('l_song'))
        try: 
            epocasmlp=int(request.POST.get('epoch_train'))
        except:
            epocasmlp=None
        for f in files:
            fs=FileSystemStorage()
            name = fs.save(f.name,f)
            fs.path(name)
            midiList.append(fs.path(name))
        #DT_RF_MLP_free_windows(midiList,division_tiempo_compas,size_win,duracion_pista,path_exit_file,numero_de_canciones_a_generar,modelo,numero_epocas)
        DT_RF_MLP_free_windows(midiList,division_tiempo_compas,size_win,long_canciones,settings.MEDIA_ROOT,numero_canciones,modelo,epocasmlp)
        for f in midiList:
            fs.delete(f)
        for i in range(numero_canciones+1):
            dic[str(i)]=settings.MEDIA_URL+"{}.mid".format(i)
    return render(request, "core/uno.html",context={"dic":dic})

def dos(request):
    dic = {}
    midiList=[]
    if request.method == 'POST':
        files=request.FILES.getlist('midi')
        numero_canciones=int(request.POST.get('n_song'))
        modelo=request.POST.get('gender')
        division_tiempo_compas=int(request.POST.get('division'))
        entrenar_por=request.POST.get('tipoentrenamiento')
        long_canciones=int(request.POST.get('l_song'))
        try: 
            epocasmlp=int(request.POST.get('epoch_train'))
        except:
            epocasmlp=None
        for f in files:
            fs=FileSystemStorage()
            name = fs.save(f.name,f)
            fs.path(name)
            midiList.append(fs.path(name))
        #DT_RF_MLP(midiList,division_tiempo_compas,entrenar_por,path_exit_file,duracion_pista,modelo,numero_de_canciones_a_generar,numero_epocas)
        DT_RF_MLP(midiList,division_tiempo_compas,entrenar_por,settings.MEDIA_ROOT,long_canciones,modelo,numero_canciones,epocasmlp)
        for f in midiList:
            fs.delete(f)
        for i in range(numero_canciones+1):
            dic[str(i)]=settings.MEDIA_URL+"{}.mid".format(i)
    return render(request, "core/dos.html",context={"dic":dic})
   

def tres(request):
    dic = {}
    midiList=[]
    if request.method == 'POST':
        files=request.FILES.getlist('midi')
        numero_canciones=int(request.POST.get('n_song'))
        division_tiempo_compas=int(request.POST.get('division'))
        size_win=int(request.POST.get('sizew'))
        long_canciones=int(request.POST.get('l_song'))
        epocasmlp=int(request.POST.get('epoch_train'))
        for f in files:
            fs=FileSystemStorage()
            name = fs.save(f.name,f)
            fs.path(name)
            midiList.append(fs.path(name))
        #LSTM_free_windows(midiList,division_tiempo_compas,size_win,duracion_pista,path_exit_file,numero_de_canciones_a_generar,numero_epocas)
        LSTM_free_windows(midiList,division_tiempo_compas,size_win,long_canciones,settings.MEDIA_ROOT,numero_canciones,epocasmlp)
        for f in midiList:
            fs.delete(f)
        for i in range(numero_canciones+1):
            dic[str(i)]=settings.MEDIA_URL+"{}.mid".format(i)
    return render(request, "core/tres.html",context={"dic":dic})
  

def cuatro(request):
    dic = {}
    midiList=[]
    if request.method == 'POST':
        files=request.FILES.getlist('midi')
        numero_canciones=int(request.POST.get('n_song'))
        division_tiempo_compas=int(request.POST.get('division'))
        entrenar_por=request.POST.get('tipoentrenamiento')
        long_canciones=int(request.POST.get('l_song'))
        epocasmlp=int(request.POST.get('epoch_train'))
        for f in files:
            fs=FileSystemStorage()
            name = fs.save(f.name,f)
            fs.path(name)
            midiList.append(fs.path(name))
        #LSTM_custom(path_open_file,division_tiempo_compas,entrenar_por,path_exit_file,duracion_pista,numero_de_canciones_a_generar,numero_epocas)
        LSTM_custom(midiList,division_tiempo_compas,entrenar_por,settings.MEDIA_ROOT,long_canciones,numero_canciones,epocasmlp)
        for f in midiList:
            fs.delete(f)
        for i in range(numero_canciones+1):
            dic[str(i)]=settings.MEDIA_URL+"{}.mid".format(i)
    return render(request, "core/cuatro.html",context={"dic":dic})


def cinco(request):
    dic = {}
    midiList=[]
    if request.method == 'POST':
        files=request.FILES.getlist('midi')
        numero_canciones=int(request.POST.get('n_song'))
        division_tiempo_compas=int(request.POST.get('division'))
        size_win=int(request.POST.get('sizew'))
        long_canciones=int(request.POST.get('l_song'))
        epocasmlp=int(request.POST.get('epoch_train'))
        for f in files:
            fs=FileSystemStorage()
            name = fs.save(f.name,f)
            fs.path(name)
            midiList.append(fs.path(name))
        #CNN_free_windows(midiList,division_tiempo_compas,size_win,duracion_pista,path_exit_file,numero_de_canciones_a_generar,numero_epocas)
        CNN_free_windows(midiList,division_tiempo_compas,size_win,long_canciones,settings.MEDIA_ROOT,numero_canciones,epocasmlp)
        for f in midiList:
            fs.delete(f)
        for i in range(numero_canciones+1):
            dic[str(i)]=settings.MEDIA_URL+"{}.mid".format(i)
    return render(request, "core/cinco.html",context={"dic":dic})

def seis(request):
    dic = {}
    midiList=[]
    if request.method == 'POST':
        files=request.FILES.getlist('midi')
        numero_canciones=int(request.POST.get('n_song'))
        division_tiempo_compas=int(request.POST.get('division'))
        entrenar_por=request.POST.get('tipoentrenamiento')
        long_canciones=int(request.POST.get('l_song'))
        epocasmlp=int(request.POST.get('epoch_train'))
        for f in files:
            fs=FileSystemStorage()
            name = fs.save(f.name,f)
            fs.path(name)
            midiList.append(fs.path(name))
        #CNN_custom(midiList,division_tiempo_compas,entrenar_por,path_exit_file,duracion_pista,numero_de_canciones_a_generar,numero_epocas)
        CNN_custom(midiList,division_tiempo_compas,entrenar_por,settings.MEDIA_ROOT,long_canciones,numero_canciones,epocasmlp)
        for f in midiList:
            fs.delete(f)
        for i in range(numero_canciones+1):
            dic[str(i)]=settings.MEDIA_URL+"{}.mid".format(i)
    return render(request, "core/seis.html",context={"dic":dic})

def siete(request):
    return render(request,"core/siete.html")

























