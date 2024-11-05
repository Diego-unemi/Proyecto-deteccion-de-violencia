
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
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
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('modulos/', views.modulos_view, name='modulos'),
    path('module/', views.module_select, name='module'),
    path('video_upload/', views.video_upload, name='video_upload'),
    path('importar_video/', views.analizar_video_importado, name='importar_video'),
    path('incident-search-ajax/', views.incident_search_ajax, name='incident_search_ajax'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)