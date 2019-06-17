from django.db import models


class Account(models.Model):
    title_choice = [
        ('C', 'C-Ops'),
        ('M', 'Manager'),
        ('AS', 'SME'),
        ('AC', 'C-Ops & Auditor'),
        ('D', 'Developer'),
        ('AD', 'Advanced User')
    ]
    auditor_login = models.CharField(max_length =30,null=False,unique=True)
    first_name = models.CharField('First Name',max_length=30)
    last_name = models.CharField('Last Name',max_length=30)
    job_title = models.CharField(verbose_name='Job Title', max_length=15,choices=title_choice)

    def __repr__(self):
        return self.auditor_login





