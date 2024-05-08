from django.urls import path
from . import views


urlpatterns = [
    path('', views.index),
    path('index/', views.index),
    path('algorithms/', views.algorithms),
    path('makecrimeanalysis/', views.makecrimeanalysis),
    path('blog/', views.blog),
    path('about/', views.about),
]
