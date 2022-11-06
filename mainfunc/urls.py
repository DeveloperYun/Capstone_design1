from django.urls import path, re_path
from . import views

app_name = 'mainfunc'

urlpatterns = [
    path('main/',  views.main, name='main'),
    path('main2/', views.main2, name='main2'),
    path('main3/', views.main3, name='main3'),
    path('main4/', views.main4, name='main4'),
]