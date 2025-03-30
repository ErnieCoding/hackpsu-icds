from django.urls import path
from . import views

urlspatterns =[
    path('process-point-cloud/', views.process_point_cloud, name="process_point_cloud"),
]