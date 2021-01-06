from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('signup', views.handleSignup, name='handleSignup'),
    path('login', views.handleLogin, name='handleLogin'),
    path('logout', views.handleLogout, name='handleLogout'),
    path('log', views.handleLog, name='handleLog'),
    path('sign', views.handleSign, name='handleSign'),
    path('aboutus',views.about,name='about'),
    path('sum',views.logsum,name='logsum'),
    path('gensum', views.generatesum, name='generatesum'),
    path('humansum', views.humansum, name='humansum'),
    path('savehumansum', views.savehumansum, name='savehumansum'),

]