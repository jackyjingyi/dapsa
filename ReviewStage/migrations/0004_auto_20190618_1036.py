# Generated by Django 2.1.7 on 2019-06-18 10:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ReviewStage', '0003_auto_20190614_1614'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='speclevel',
            name='incorrect_test_method',
        ),
        migrations.RemoveField(
            model_name='speclevel',
            name='issue_found',
        ),
        migrations.RemoveField(
            model_name='speclevel',
            name='report_number',
        ),
        migrations.RemoveField(
            model_name='speclevel',
            name='safety_and_regulatory',
        ),
        migrations.RemoveField(
            model_name='speclevel',
            name='speck_number',
        ),
        migrations.AddField(
            model_name='speclevel',
            name='addition',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='audit_date',
            field=models.DateField(auto_now=True),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='auditor',
            field=models.CharField(max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='description',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='issue',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='method',
            field=models.CharField(max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='mfp',
            field=models.CharField(max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='protocol',
            field=models.CharField(max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='remark',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='section',
            field=models.CharField(max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='speclevel',
            name='sr',
            field=models.CharField(max_length=200, null=True),
        ),
    ]
