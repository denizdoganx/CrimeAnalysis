from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse

# Create your views here.
def index(request):
    return render(request, "pages/index.html")


def blog(request):
    return render(request, "pages/blog.html")


def about(request):
    return render(request, "pages/about.html")


def algorithms(request):
    redirect_url = reverse('algorithms_index_page_link', args=[])
    return HttpResponseRedirect(redirect_url)


def makecrimeanalysis(request):
    redirect_url = reverse('make_crime_analysis_index_page_link', args=[])
    return HttpResponseRedirect(redirect_url)
