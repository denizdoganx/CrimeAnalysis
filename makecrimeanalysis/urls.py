from django.urls import path
from . import views


urlpatterns = [
    path('index/', views.index, name="make_crime_analysis_index_page_link")
]
