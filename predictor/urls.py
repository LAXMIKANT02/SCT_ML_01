from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('history/', views.prediction_history, name='history'),
    path('delete/<int:pk>/', views.delete_prediction, name='delete_prediction'),
    path('delete_all/', views.delete_all_predictions, name='delete_all_predictions'),
]
