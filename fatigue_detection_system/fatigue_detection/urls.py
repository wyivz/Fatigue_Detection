"""
URL configuration for fatigue_detection project.
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
    path('', RedirectView.as_view(url='/detection/dashboard/')),
    path('users/login/', RedirectView.as_view(url='/accounts/login/')),
    path('users/', RedirectView.as_view(url='/accounts/')),
]

# Local packaged app: serve media + project static even when DEBUG=0
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
if settings.STATICFILES_DIRS:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
elif getattr(settings, "STATIC_ROOT", None):
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static('/webfonts/', document_root=settings.BASE_DIR / 'webfonts')
