from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    # 仪表盘
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # 检测功能
    path('realtime/', views.realtime_detection, name='realtime'),
    path('video/', views.video_detection, name='video'),
    
    # 检测历史
    path('history/', views.history, name='history'),
    
    # 检测过程和结果
    path('session/<int:session_id>/', views.session_detail, name='session_detail'),
    path('results/<int:session_id>/', views.detection_results, name='detection_results'),
    
    # API端点（用于AJAX请求）
    path('api/start_detection/', views.start_detection, name='start_detection'),
    path('api/stop_detection/', views.stop_detection, name='stop_detection'),
    path('api/get_result/', views.get_result, name='get_result'),
    
    # 统计分析
    path('statistics/', views.statistics, name='statistics'),
    
    # 系统配置（仅管理员）
    path('config/', views.system_config, name='system_config'),
] 