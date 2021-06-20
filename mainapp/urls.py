from django.urls import path
from . import views
from django.contrib.auth.views import LoginView, LogoutView

urlpatterns = [
    path('', views.index, name='index'),
    path('rust', views.indexrust, name='indexrust'),
    path('anpr', views.indexanpr, name='indexanpr'),
    path('scratch', views.indexscratch, name='indexscratch'),
    path('analyze/', views.analyze, name='analyze_view'),
    path('analyze/rust', views.analyzerust, name='analyze_rust'),
    path('analyze/anpr', views.analyzeanpr, name='analyze_anpr'),
    path('analyze/scratch', views.analyzescratch, name='analyze_scratch'),
    path('account/login/', LoginView.as_view(template_name='index/login.html'), name='login'),
    path('account/signup/', views.signup_view, name='signup'),
    path('account/logout/', LogoutView.as_view(), name='logout'),
]