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

            print("Video path", video_path)
            result = predict(video_path)
            return render(request,"result.html",{"result": result, "video": video})
    else:
        form = VideoForm()
    return render(request,'index.html',{"form":form})

