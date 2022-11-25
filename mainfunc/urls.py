from django.urls import path, re_path
from . import views

app_name = 'mainfunc'

urlpatterns = [
    path('main/',  views.main, name='main'),
    path('main2/', views.main2, name='main2'),
    path('main3/', views.main3, name='main3'),
    path('main4/', views.main4, name='main4'),
    path('tip/', views.tip, name='tip'),
    path('tip2/', views.tip2, name='tip2'),
    path('tip3/', views.tip3, name='tip3'),
    path('tip4/', views.tip4, name='tip4'),
]