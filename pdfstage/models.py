from django.db import models


# store a reference to the actually file to the db
# core function will deal with the files
#
class Document(models.Model):
    description = models.CharField(max_length=255,blank=True)
    document = models.FileField(upload_to='documents/%Y/%m/%d')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    analyzed = models.BooleanField(default=False)

    def analyze(self):
        self.analyzed = True
        return

    def __repr__(self):
        return ("<description: %s ,\n" 
               "document: %s, \n" 
               "uploaded_at: %s, \n" 
               "analyzed: %s.>\n") %(self.description,self.document,self.uploaded_at,self.analyzed)



