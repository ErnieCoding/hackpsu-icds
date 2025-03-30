from django.urls import path
from . import process_visual

urlpatterns = [
    path('api/process-point-cloud/', process_visual.process_point_cloud, name='process_point_cloud'),
]