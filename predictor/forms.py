from django import forms

class HouseForm(forms.Form):
    sq_footage = forms.FloatField(label='Square Footage', min_value=100)
    bedrooms = forms.IntegerField(label='Bedrooms', min_value=0)
    bathrooms = forms.FloatField(label='Bathrooms', min_value=0.0)
class UploadCSVForm(forms.Form):
    csv_file = forms.FileField(label="Upload Test CSV")