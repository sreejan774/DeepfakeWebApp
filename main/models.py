from django.db import models
from .validators import FileSizeValidator

# Create your models here.
class Video(models.Model):
    video = models.FileField(upload_to="video/", validators=[FileSizeValidator])
    def __str__(self):
        return self.video.name

    def getName(self):
        return self.video.name