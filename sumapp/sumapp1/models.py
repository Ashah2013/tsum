from django.db import models
from django.contrib.auth.models import User
from datetime import datetime  

class TextInfo(models.Model):
    otext=models.TextField(max_length='6000')
    stextrank = models.TextField(max_length='6000')
    stfidf = models.TextField(max_length='6000')
    shumansum = models.TextField(max_length='6000', null=True)
    uid=models.EmailField()
