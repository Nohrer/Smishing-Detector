from django.db import models

class Message(models.Model):
    text = models.TextField()
    result = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)
