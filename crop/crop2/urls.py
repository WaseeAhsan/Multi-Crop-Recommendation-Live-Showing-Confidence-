from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),  # The home page URL
    path('recommend/', views.crop_recommendation, name="recommend"),  # The recommendation URL
]
