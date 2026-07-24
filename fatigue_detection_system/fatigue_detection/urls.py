"""
URL configuration for fatigue_detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls', namespace='accounts')),
    path('detection/', include('detection.urls', namespace='detection')),
    path('', RedirectView.as_view(url='/detection/dashboard/')),  # 重定向根URL到仪表盘
    
    # 添加兼容性重定向
    path('users/login/', RedirectView.as_view(url='/accounts/login/')),
    path('users/', RedirectView.as_view(url='/accounts/')),
]

# 在开发环境中提供媒体文件
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # 添加对webfonts目录的静态文件服务
    urlpatterns += static('/webfonts/', document_root=settings.BASE_DIR / 'webfonts')
