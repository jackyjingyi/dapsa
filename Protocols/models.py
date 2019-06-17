from django.db import models
import csv
import datetime

loc = 'C:/Users/yanjingy/Documents/work/protocol/TUV protocols/'
file_name = 'AMZ-034-WW_One Time Use Tableware_V2.0_20180629.csv'


class Protocolid(models.Model):
    name = models.TextField(unique=True,verbose_name='Protocol Name')
    life = models.CharField(max_length=30)


class Protocol(models.Model):
    name = models.ForeignKey('Protocolid',verbose_name='Protocol Name',on_delete=models.CASCADE,default='')
    speck_number = models.CharField(verbose_name='Speck Number', null=True, default='',max_length=30)
    regulation = models.TextField(verbose_name='Regulation')
    requirement_title = models.TextField(verbose_name='Requirement title', db_column='test item')
    link = models.URLField(max_length=200)
    region = models.CharField(verbose_name='Region', max_length=100)
    test_method = models.TextField(verbose_name='Test Method')
    requirement = models.TextField(verbose_name='Requirement')
    protduct_scope = models.CharField(verbose_name='Product Scope',
                                      max_length=100,
                                      db_column='scope')
    exemption = models.CharField(max_length=30, default='NA')
    protocol_section = models.CharField(max_length=100, null=False,db_column='section')
    mandatory_voluntary = models.CharField(max_length=30, db_column='restrict')
    is_cornerstone = models.BooleanField(verbose_name='Is Cornerstone', default=False, db_column='cornerstone')
    new_voluntary_safety_standard = models.TextField(verbose_name='New Voluntary Safety Standard',
                                                     db_column='new_restrict',
                                                     null=True)
    reationale = models.CharField(verbose_name='Reationale',
                                  max_length=50, null=True)
    upload_date = models.DateField(auto_now_add=True)


    @classmethod
    def bulk_insert_from_csv(cls,loc,file_name):
        with open(loc + file_name, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                    newp = Protocol(id=row[0], name=Protocolid.objects.get(name=row[1],life='Active'),
                             speck_number=row[2],
                             regulation=row[3],
                             requirement_title=row[4],
                             link = row[5],
                             region=row[6],
                             test_method=row[7],
                             requirement=row[8],
                             protduct_scope=row[9],
                             exemption=row[10],
                             protocol_section=row[11],
                             mandatory_voluntary=row[12],
                             is_cornerstone=row[13],
                             new_voluntary_safety_standard=row[14],
                             reationale=row[15],
                             upload_date= datetime.datetime.today())

                    newp.save()

