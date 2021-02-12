from django.urls import path
from .  import views

urlpatterns = [

    path('', views.home,name="home"),
    path('uno/', views.uno,name="uno"),
    path('dos/', views.dos,name="dos"),
    path('tres/', views.tres,name="tres"),
    path('cuatro/', views.cuatro,name="cuatro"),
    path('cinco/', views.cinco,name="cinco"),
    path('seis/', views.seis,name="seis"),
    path('siete/', views.siete,name="siete"),
   
    
]