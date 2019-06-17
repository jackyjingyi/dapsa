import django_tables2 as tables
from .models import Protocol,Protocolid


class IndexTable(tables.Table):
    class Meta:
        model = Protocolid
        template_name = 'django_tables2/bootstrap.html'


class ProtocolTable(tables.Table):
    class Meta:
        model = Protocol
        template_name = 'django_tables2/bootstrap.html'



