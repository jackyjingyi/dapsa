from django import forms
from pdfstage.models import Document
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout,Div,Submit,Row,Column,Field



class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('description', 'document',)

