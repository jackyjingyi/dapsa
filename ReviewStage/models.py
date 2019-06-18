import csv
from django.db import models
from SamplingStage.models import MFP
from auditors.models import Account
import datetime

loc = 'C:/Users/yanjingy/Documents/work/protocol/TUV protocols/'
file_name = 'status.csv'
DefaultDate ='1991-10-23'

class ReportInfo(models.Model):
    lab_list =[
        ('BV', 'BV'),
        ('ITS', 'ITS'),
        ('TUV', 'TUV'),
    ]
    auditor_id = models.ForeignKey(Account,on_delete=models.CASCADE)
    audit_date = models.DateField(auto_now_add=True)
    mfp = models.ForeignKey(MFP, on_delete=models.CASCADE)
    report_number = models.TextField(verbose_name='Report Number')
    protocol_on_report = models.TextField(verbose_name='Protocol is ')
    report_issue_date = models.DateField(default=DefaultDate)
    lab = models.CharField(choices=lab_list, max_length=30)
    pdf_name = models.TextField(verbose_name='PDF name is')


class IssuePool(models.Model):
    issue = models.TextField(verbose_name='Issue Found is')
    description = models.TextField(verbose_name='Related Scenario')
    setup_date = models.DateField(default=DefaultDate)
    setup_by = models.ForeignKey(Account, verbose_name='Setup by', on_delete=models.CASCADE, default='')


class StatusAndNextStep(models.Model):
    """
    ASIN level
    """
    short_cut = models.CharField(verbose_name='In short',max_length=50,default='django flying')
    status = models.TextField(verbose_name='Status',  null=False)
    next_step = models.TextField(verbose_name='Next Step', null=False, db_column='next_step')
    status_description = models.TextField(verbose_name='Status Description', db_column= 'status_de',null=True)
    next_step_description = models.TextField(verbose_name='Next Step Description', db_column='next_de',null=True)
    setup_date = models.DateField(verbose_name='Setup Date', default=DefaultDate)
    setup_by = models.ForeignKey(Account, on_delete=models.CASCADE,default='')

    @classmethod
    def bulk_insert_from_csv(cls,loc, file_name, header =True):
        with open(loc + file_name, 'r') as csv_file:
            reader = csv.reader(csv_file)
            if header:
                next(reader)
            for row in reader:
                news = StatusAndNextStep(
                    status = row[0],
                    next_step= row[1],
                    short_cut= row[2],
                    status_description= row[3],
                    next_step_description=row[4],
                    setup_date=row[5],
                    setup_by= Account.objects.get(auditor_login=row[6])
                )
                news.save()


class SpecLevel(models.Model):
    auditor = models.CharField(max_length=50,null=True)
    protocol = models.CharField(max_length=200,null=True)
    mfp = models.CharField(max_length=200,null=True)
    section = models.CharField(max_length=200,null=True)
    description = models.TextField(default='')
    method = models.CharField(max_length=200,null=True)
    issue = models.TextField(default='')
    sr = models.CharField(max_length=200,null=True)
    addition = models.TextField(default='')
    remark = models.TextField(default='')
    audit_date = models.DateField(auto_now=True)


