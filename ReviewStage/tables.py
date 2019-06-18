import django_tables2 as tables
from .models import SpecLevel


class Spec_table(tables.Table):
    class Meta:
        model = SpecLevel
        template_name = 'django_tables2/bootstrap.html'