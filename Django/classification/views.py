from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.generic.base import TemplateView
from .inference.model_inference import classify
from django.conf import settings
from pathlib import Path
from .forms import UploadImageForm
from django.views import generic

# Create your views here.


def home_view(request):
    return render(request, 'home.html')
    

def classificaion_view(request):
    form = UploadImageForm(request.POST or None, request.FILES)
    result = None
    if form.is_valid():
        image_field = form.cleaned_data['image']
        form.save()
        result = classify(settings.MEDIA_ROOT, image_field.name)
    context = {
        'form':form,
        'result':result
    }
    return render(request, 'classificaion.html', context)