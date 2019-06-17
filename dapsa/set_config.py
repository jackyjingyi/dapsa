import pandas as pd
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice,PDFTextDevice
from pdfminer.layout import LAParams,LTChar,LTTextLine,LTText,LTTextBox,LTLine,LTRect,LTImage,LTTextBoxHorizontal
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import Plane
import pdfminer
import re
import copy
import openpyxl
import time
import datetime
import nltk

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from pdfstage.models import Document



b = Document.objects.get(pk=1)
print(b.__repr__())
b.analyze()
b.save()
print(b.__repr__())
m = Document.objects.order_by('-uploaded_at')
print(m.__repr__())
print(b.document.path)

