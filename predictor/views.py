from django.shortcuts import render
from .forms import HouseForm
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join('data', 'house_price_model.pkl')

# Load model once when app starts
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def home(request):
    form = HouseForm()
    return render(request, 'predictor/index.html', {'form': form})

def predict(request):
    if request.method == 'POST':
        form = HouseForm(request.POST)
        if form.is_valid():
            sqft = form.cleaned_data['sq_footage']
            bed = form.cleaned_data['bedrooms']
            bath = form.cleaned_data['bathrooms']

            features = np.array([[sqft, bed, bath]])
            prediction = model.predict(features)[0]

            return render(request, 'predictor/result.html', {
                'prediction': round(prediction, 2),
                'form': form
            })
    else:
        form = HouseForm()
    return render(request, 'predictor/index.html', {'form': form})
