from django.contrib import admin
from django.urls import path, include
from hello import views
from django.conf.urls.static import static

# from hello import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', include('hello.urls')),
    path('accounts/', include('accounts.urls')),
    path('mainfunc/', include('mainfunc.urls')),
    path('', views.index, name='index'),
]
