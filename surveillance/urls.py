
from django.urls import path
from . import views
urlpatterns = [
    path('', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('contact/', views.contact_view, name='contact'),
    path('camera/', views.camera_view, name='camera'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('incident_list/', views.incident_list, name='incident_list'),
    path('stop_feed/', views.stop_feed, name='stop_feed'),
    path('delete_incident/<int:id>/', views.delete_incident, name='delete_incident'),
]
