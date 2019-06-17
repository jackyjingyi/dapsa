from django.shortcuts import render
from django.http import HttpResponse
from django_tables2 import RequestConfig
from .models import Protocol, Protocolid
from .tables import ProtocolTable,IndexTable


def index(request):
    table = IndexTable(Protocolid.objects.all())
    RequestConfig(request).configure(table)
    return render(request,'Protocols/index.html', {'table':table})


def protocols_display(request):
    table = ProtocolTable(Protocol.objects.all())
    RequestConfig(request).configure(table)
    return render(request,'Protocols/protocols_display.html', {'table':table})