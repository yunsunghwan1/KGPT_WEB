from django.contrib import admin
from django.urls import path, include
from GPTapp import views 
urlpatterns = [
    path('', views.index),
    path('GPT/', include('GPTapp.urls')),
    path('admin/', admin.site.urls),
]
