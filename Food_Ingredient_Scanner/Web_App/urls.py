from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('ingredient-scanner/', views.ingredient_scanner, name='ingredient_scanner'),
    path('about/', views.about, name='about'),
    path('features/', views.features, name='features'),
    path('contact/', views.contact, name='contact'),
    path('upload-success/', views.upload_success, name='upload_success'),
    path('analyse-ingredients/', views.analyse_ingredients, name='analyse_ingredients'),
    path('generate-video/', views.generate_video, name='generate_video')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)