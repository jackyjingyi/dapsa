from django.urls import path
from . import views

urlpatterns = [

    path('display/', views.spec_level_display, name = 'Spec_Level'),
]