from django.urls import path
from . import process_visual

urlpatterns = [
    path('api/process-point-cloud/', process_visual.process_point_cloud, name='process_point_cloud'),
    path('api/view-point-cloud/', process_visual.view_point_cloud, name='view_point_cloud'),
    path('api/run-detection/', process_visual.run_object_detection, name='run_object_detection'),
]