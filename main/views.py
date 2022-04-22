from unittest import result
from django.http import HttpResponse
from django.shortcuts import render
from .forms import VideoForm
from .deepfake_pipeline.deepfake_predicion_pipeline import predict
import os


def index(request):
    if request.method == "POST":
        form = VideoForm(data=request.POST,files=request.FILES)
        if form.is_valid():
            form.save()
            video = form.instance
            video_name = video.getName()
            media_path = os.path.join(os.getcwd(),'media')
            video_path = os.path.join(media_path,video_name)
            score = predict(video_path)
            if score > 0.5:
                result = "real"
            else:
                result = "fake"
            return render(request,"result.html",{"result": result,"score":score})
    else:
        form = VideoForm()
    return render(request,'index.html',{"form":form})

