from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from .forms import HouseForm
from .models import PredictionHistory
from django.conf import settings
import joblib
import numpy as np
import os

model_path = os.path.join(settings.BASE_DIR, 'data/linear_model.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'data/scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def home(request):
    form = HouseForm()
    return render(request, 'predictor/html/index.html', {'form': form})

def predict(request):
    if request.method == 'POST':
        form = HouseForm(request.POST)
        if form.is_valid():
            sqft = form.cleaned_data['sq_footage']
            bed = form.cleaned_data['bedrooms']
            bath = form.cleaned_data['bathrooms']

            features = np.array([[sqft, bed, bath]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            PredictionHistory.objects.create(
                square_footage=sqft,
                bedrooms=bed,
                bathrooms=bath,
                predicted_price=prediction
            )

            return render(request, 'predictor/html/result.html', {
                'prediction': round(prediction, 2),
                'form': form
            })
    else:
        form = HouseForm()
    return render(request, 'predictor/html/index.html', {'form': form})

def prediction_history(request):
    records = PredictionHistory.objects.all().order_by('-created_at')
    return render(request, 'predictor/html/history.html', {'records': records})

@require_POST
def delete_prediction(request, pk):
    try:
        record = PredictionHistory.objects.get(pk=pk)
        record.delete()
        return JsonResponse({'status': 'success'})
    except PredictionHistory.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Record not found'}, status=404)

@require_POST
def delete_all_predictions(request):
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        PredictionHistory.objects.all().delete()
        return JsonResponse({'status': 'all_deleted'})
    return HttpResponseBadRequest('Invalid request')
