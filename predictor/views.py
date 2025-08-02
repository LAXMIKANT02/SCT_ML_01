from django.shortcuts import render
from .forms import HouseForm
from .models import PredictionHistory
import joblib
from django.conf import settings
import numpy as np
import os


model_path = os.path.join(settings.BASE_DIR, 'data/linear_model.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'data/scaler.pkl')

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def home(request):
    form = HouseForm()
    return render(request, 'predictor/index.html', {'form': form})

def predict(request):
    if model is None or scaler is None:
        return render(request, 'predictor/error.html', {
            'message': 'Model or Scaler is not loaded. Please contact admin.'
        })

    if request.method == 'POST':
        form = HouseForm(request.POST)
        if form.is_valid():
            sqft = form.cleaned_data['sq_footage']
            bed = form.cleaned_data['bedrooms']
            bath = form.cleaned_data['bathrooms']

            features = np.array([[sqft, bed, bath]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]


            # Save to DB
            PredictionHistory.objects.create(
                square_footage=sqft,
                bedrooms=bed,
                bathrooms=bath,
                predicted_price=prediction
     )

            return render(request, 'predictor/result.html', {
                'prediction': round(prediction, 2),
                'form': form
            })
    else:
        form = HouseForm()
    return render(request, 'predictor/index.html', {'form': form})

def prediction_history(request):
    records = PredictionHistory.objects.all().order_by('-created_at')
    return render(request, 'predictor/history.html', {'records': records})
