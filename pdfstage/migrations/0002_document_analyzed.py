# Generated by Django 2.1.7 on 2019-06-17 15:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pdfstage', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='analyzed',
            field=models.BooleanField(default=False),
        ),
    ]
