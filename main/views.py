from django.shortcuts import render
from .forms import VideoForm


def index(request):
    if request.method == "POST":
        form = VideoForm(data=request.POST,files=request.FILES)
        if form.is_valid():
            form.save()
            video = form.instance
            print("Video Name", video.getName())
            result = make_prediction()
            return render(request,"result.html",{"result": result, "video": video})
    else:
        form = VideoForm()
    return render(request,'index.html',{"form":form})

