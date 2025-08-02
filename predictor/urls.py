app_name = 'predictor'

from django.shortcuts import render
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('history/', views.prediction_history, name='history'),
]

def home(request):
    return render(request, 'predictor/index.html')
