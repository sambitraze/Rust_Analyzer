from django.urls import path
from . import views
from django.contrib.auth.views import LoginView, LogoutView

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze, name='analyze_view'),
    path('account/login/', LoginView.as_view(template_name='index/login.html'), name='login'),
    path('account/signup/', views.signup_view, name='signup'),
    path('account/logout/', LogoutView.as_view(), name='logout'),
]