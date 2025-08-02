from django.db import models

class PredictionHistory(models.Model):
    square_footage = models.FloatField()
    bedrooms = models.IntegerField()
    bathrooms = models.IntegerField()
    predicted_price = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.square_footage} sqft, {self.bedrooms} bed, {self.bathrooms} bath"
