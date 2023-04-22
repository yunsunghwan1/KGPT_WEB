from django.contrib import admin
from django.urls import path

from . import views
urlpatterns = [
    path('', views.index),
    # path('', views.generate_text, name='generate_text'),
    path('input/', views.KGPT_input),
    path('start/', views.generate_text),
    path('chatbot/', views.generate_text),
]
