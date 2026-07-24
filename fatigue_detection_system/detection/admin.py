from django.contrib import admin
from .models import DetectionSession, DetectionResult, SystemConfig

@admin.register(DetectionSession)
class DetectionSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'session_type', 'start_time', 'end_time', 'get_detection_count')
    list_filter = ('session_type', 'start_time')
    search_fields = ('user__username',)
    date_hierarchy = 'start_time'

@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'timestamp', 'face_detected', 'smoking_detected', 
                   'phone_detected', 'drinking_detected', 'yawn_detected', 'fatigue_level')
    list_filter = ('face_detected', 'smoking_detected', 'phone_detected', 
                  'drinking_detected', 'yawn_detected', 'fatigue_level')
    search_fields = ('session__user__username',)
    date_hierarchy = 'timestamp'

@admin.register(SystemConfig)
class SystemConfigAdmin(admin.ModelAdmin):
    list_display = ('config_key', 'config_value', 'updated_at')
    search_fields = ('config_key', 'description')
    date_hierarchy = 'updated_at'
