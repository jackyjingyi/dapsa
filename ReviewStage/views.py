from django.shortcuts import render
from .models import SpecLevel
from .tables import Spec_table
from django_tables2 import RequestConfig

def spec_level_display(request):
    table =Spec_table(SpecLevel.objects.all())
    RequestConfig(request).configure(table)
    return render(request,'ReviewStage/spec_level_display.html', {'table':table})