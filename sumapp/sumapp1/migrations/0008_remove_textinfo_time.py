# Generated by Django 2.1.7 on 2021-01-05 22:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sumapp1', '0007_auto_20210106_0328'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='textinfo',
            name='time',
        ),
    ]
