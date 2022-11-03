from django.shortcuts import render, HttpResponse
import random
# Create your views here.


def index(request):
    return HttpResponse('<h1>T1</h1>'+str(random.random()))


def create(req):
    return HttpResponse('흐애!!')


def read(re, id):
    return HttpResponse('Faker coming back!!!'+id)
