from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'accounts'

urlpatterns = [
    # 登录和登出
    path('login/', views.user_login, name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    
    # 注册
    path('register/', views.user_register, name='register'),
    
    # 个人资料
    path('profile/', views.user_profile, name='profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
    path('profile/update/', views.update_profile, name='update_profile'),
    
    # 修改密码
    path('password_change/', auth_views.PasswordChangeView.as_view(
        template_name='accounts/password_change.html',
        success_url='/accounts/password_change_done/'),
        name='password_change'),
    path('password_change_done/', auth_views.PasswordChangeDoneView.as_view(
        template_name='accounts/password_change_done.html'),
        name='password_change_done'),
] 