import pandas as pd
import re
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from pdfstage.models import Document



a = Document.objects.all()
print(a)