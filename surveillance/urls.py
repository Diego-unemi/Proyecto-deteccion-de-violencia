
from django.urls import path
from . import views
urlpatterns = [
    path('', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('contact/', views.contact_view, name='contact'),
    path('camera/', views.camera_view, name='camera'),
    path('video_feed/', views.video_feed_view, name='video_feed'),

    
]
