from django.shortcuts import render
import os
import json
import cv2
import base64
import numpy as np
from datetime import datetime, timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib import messages
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils import timezone

from .models import DetectionSession, DetectionResult, SystemConfig
from .utils.yolo_detector import YOLODetector
from .utils.dlib_detector import DlibDetector

import logging
logger = logging.getLogger("detection.detect")

def _safe_log(msg):
    """Route through the logging config (rotating file under logs/) instead
    of print(), which can raise OSError: [Errno 22] if Windows stdout is
    detached (e.g. running via pythonw / a service)."""
    try:
        logger.info(msg)
    except Exception:
        pass

# Browser-path persist throttle (session_id -> last persist unix time / fatigue level)
_browser_last_persist_at = {}
_browser_last_fatigue_level = {}
_browser_last_yawn = {}


def _resize_for_detect(image, configs=None):
    """Optionally shrink frame to yolo_detect_max_width (0 = no shrink, archive-like)."""
    if image is None:
        return image
    if configs is None:
        try:
            from .utils.config_cache import get_configs

            configs = get_configs()
        except Exception:  # noqa: BLE001
            configs = {}
    try:
        raw = (configs or {}).get("yolo_detect_max_width", 0)
        max_w = int(float(raw if raw is not None and str(raw).strip() != "" else 0))
    except (TypeError, ValueError):
        max_w = 0
    if max_w <= 0:
        return image
    max_w = max(640, min(4096, max_w))
    h0, w0 = image.shape[:2]
    if w0 <= max_w:
        return image
    nh = int(h0 * (max_w / float(w0)))
    return cv2.resize(image, (max_w, nh), interpolation=cv2.INTER_AREA)

# 检查权重文件是否存在
yolo_weights_path = os.path.join(settings.BASE_DIR, 'weights', 'best.pt')
dlib_weights_path = os.path.join(settings.BASE_DIR, 'weights', 'shape_predictor_68_face_landmarks.dat')

yolo_weights_exist = os.path.exists(yolo_weights_path)
dlib_weights_exist = os.path.exists(dlib_weights_path)

# 创建检测器实例
try:
    if not yolo_weights_exist:
        _safe_log(f"警告: YOLO权重文件不存在: {yolo_weights_path}")
        _safe_log("请下载权重文件并放到weights目录")
        yolo_detector = None
    else:
        yolo_detector = YOLODetector()
        _safe_log("YOLO 检测器加载成功")
except Exception as e:
    yolo_detector = None
    _safe_log(f"YOLO 检测器加载失败: {e}")

try:
    if not dlib_weights_exist:
        _safe_log(f"警告: dlib面部特征点预测器文件不存在: {dlib_weights_path}")
        _safe_log("请下载预测器文件并放到weights目录")
        dlib_detector = None
    else:
        dlib_detector = DlibDetector()
        _safe_log("dlib 检测器加载成功")
except Exception as e:
    dlib_detector = None
    _safe_log(f"dlib 检测器加载失败: {e}")

@login_required
def dashboard(request):
    """
    用户仪表盘，显示检测统计信息和最近的检测记录
    """
    # 获取用户的检测会话
    sessions = DetectionSession.objects.filter(user=request.user).order_by('-start_time')[:5]
    
    # 统计检测结果
    all_results = DetectionResult.objects.filter(session__user=request.user)
    
    # 统计各种行为的检测次数
    stats = {
        'total_sessions': sessions.count(),
        'total_detections': all_results.count(),
        'smoking_count': all_results.filter(smoking_detected=True).count(),
        'phone_count': all_results.filter(phone_detected=True).count(),
        'drinking_count': all_results.filter(drinking_detected=True).count(),
        'yawn_count': all_results.filter(yawn_detected=True).count(),
        'fatigue_count': all_results.filter(fatigue_level__gte=2).count(),
    }
    
    # 计算疲劳等级分布
    fatigue_levels = {
        'level0': all_results.filter(fatigue_level=0).count(),
        'level1': all_results.filter(fatigue_level=1).count(),
        'level2': all_results.filter(fatigue_level=2).count(),
        'level3': all_results.filter(fatigue_level=3).count(),
        'level4': all_results.filter(fatigue_level=4).count(),
    }
    
    context = {
        'sessions': sessions,
        'stats': stats,
        'fatigue_levels': fatigue_levels,
    }
    
    return render(request, 'detection/dashboard.html', context)


@login_required
@ensure_csrf_cookie
def realtime_detection(request):
    """
    实时检测页面，使用摄像头
    """
    # 创建新的检测会话
    session = DetectionSession.objects.create(
        user=request.user,
        session_type='realtime',
    )
    
    # 获取系统配置
    configs = {}
    system_configs = SystemConfig.objects.all()
    for config in system_configs:
        configs[config.config_key] = config.config_value
    
    # 设置默认配置（与 system_config 默认对齐：中度疲劳=2）
    fatigue_alert_level = int(configs.get('fatigue_alert_level', 2))
    alert_volume = int(configs.get('alert_volume', 80))
    enable_voice = configs.get('enable_voice', 'true') == 'true'
    detection_interval = int(configs.get('detection_interval', 700))
    ear_sample_interval_ms = int(configs.get('ear_sample_interval_ms', 100))
    perclos_alert_pct = float(configs.get('perclos_alert_pct', 20))
    default_source_type = configs.get('default_source_type', 'mvs')
    default_mvs_index = int(configs.get('default_mvs_index', 0))
    
    context = {
        'session': session,
        'fatigue_alert_level': fatigue_alert_level,
        'alert_volume': alert_volume,
        'enable_voice': enable_voice,
        'detection_interval': detection_interval,
        'ear_sample_interval_ms': ear_sample_interval_ms,
        'perclos_alert_pct': perclos_alert_pct,
        'default_source_type': default_source_type,
        'default_mvs_index': default_mvs_index,
    }
    
    return render(request, 'detection/realtime.html', context)

@login_required
def video_detection(request):
    """
    视频文件检测页面
    """
    if request.method == 'POST' and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        detect_fatigue = request.POST.get('detect_fatigue') == 'true'
        detect_behaviors = request.POST.get('detect_behavior') == 'true'
        
        # 创建新的检测会话
        session = DetectionSession.objects.create(
            user=request.user,
            session_type='video',
            source_file=video_file,
            status='in_progress'
        )

        from .utils.fatigue_tracker import behavior_tracker, fatigue_tracker
        configs = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
        fatigue_tracker.reset(session.id)
        behavior_tracker.reset(session.id)
        fatigue_tracker.configure(session.id, configs)
        behavior_tracker.configure(session.id, configs)
        
        try:
            # 打开视频文件进行处理
            video_path = session.source_file.path
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 每隔多少帧提取一帧进行检测
            frame_interval = 30  # 例如，每30帧检测一次
            
            frame_index = 0
            detection_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 每隔frame_interval帧进行一次检测
                if frame_index % frame_interval == 0:
                    # 处理当前帧
                    process_image(frame, session, detect_fatigue, detect_behaviors)
                    detection_count += 1
                
                frame_index += 1
                
                # 如果处理了足够的帧，就停止处理（避免过长的视频）
                if detection_count >= 20:  # 最多处理20帧
                    break
            
            # 释放视频对象
            cap.release()
            
            # 更新会话状态
            session.status = 'completed'
            session.end_time = timezone.now()
            session.save()
            fatigue_tracker.reset(session.id)
            behavior_tracker.reset(session.id)
            
            messages.success(request, f'视频处理完成，共检测了{detection_count}帧。')
            
        except Exception as e:
            # 处理出错
            session.status = 'failed'
            session.end_time = timezone.now()
            session.save()
            try:
                fatigue_tracker.reset(session.id)
                behavior_tracker.reset(session.id)
            except Exception:  # noqa: BLE001
                pass
            
            messages.error(request, f'视频处理失败: {str(e)}')
        
        return redirect('detection:session_detail', session_id=session.id)
    
    return render(request, 'detection/video_upload.html')

@login_required
def session_detail(request, session_id):
    """
    检测会话详情页面
    """
    session = get_object_or_404(DetectionSession, id=session_id, user=request.user)
    
    return render(request, 'detection/session_detail.html', {'session': session})

@login_required
def detection_results(request, session_id):
    """
    检测结果页面
    """
    session = get_object_or_404(DetectionSession, id=session_id, user=request.user)
    results = DetectionResult.objects.filter(session=session).order_by('timestamp')
    
    return render(request, 'detection/results.html', {
        'session': session,
        'results': results
    })

@login_required
def statistics(request):
    """
    统计分析页面
    """
    # 获取当前用户的所有会话
    all_sessions = DetectionSession.objects.filter(user=request.user)
    
    # 获取总会话数
    total_sessions = all_sessions.count()
    
    # 获取今日会话数
    today = timezone.now().date()
    today_sessions = all_sessions.filter(start_time__date=today).count()
    
    # 获取疲劳检测结果
    all_results = DetectionResult.objects.filter(session__user=request.user)
    
    # 获取疲劳检测次数 (疲劳等级大于等于2)
    fatigue_count = all_results.filter(fatigue_level__gte=2).count()
    
    # 获取不良行为次数
    bad_behavior_count = all_results.filter(
        smoking_detected=True
    ).count() + all_results.filter(
        phone_detected=True
    ).count() + all_results.filter(
        drinking_detected=True
    ).count() + all_results.filter(
        yawn_detected=True
    ).count()
    
    # 按会话类型统计
    realtime_count = all_sessions.filter(session_type='realtime').count()
    video_count = all_sessions.filter(session_type='video').count()
    
    # 按疲劳等级统计
    normal_count = all_results.filter(fatigue_level=0).count()
    mild_count = all_results.filter(fatigue_level=1).count()
    moderate_count = all_results.filter(fatigue_level=2).count()
    severe_count = all_results.filter(fatigue_level=3).count()
    extreme_count = all_results.filter(fatigue_level=4).count()
    
    # 按不良行为统计
    yawn_count = all_results.filter(yawn_detected=True).count()
    smoking_count = all_results.filter(smoking_detected=True).count()
    phone_count = all_results.filter(phone_detected=True).count()
    drinking_count = all_results.filter(drinking_detected=True).count()
    
    # 获取近7天的数据
    week_labels = []
    week_data = []
    
    for i in range(6, -1, -1):
        date = today - timedelta(days=i)
        week_labels.append(date.strftime('%m-%d'))
        week_data.append(all_sessions.filter(start_time__date=date).count())
    
    # 按小时段统计
    hourly_data = [0] * 8  # 8个时间段：0-3, 3-6, 6-9, 9-12, 12-15, 15-18, 18-21, 21-24
    
    for result in all_results:
        hour = result.timestamp.hour
        hourly_data[hour // 3] += 1
    
    context = {
        # 统计卡片数据
        'total_sessions': total_sessions,
        'today_sessions': today_sessions,
        'fatigue_count': fatigue_count,
        'bad_behavior_count': bad_behavior_count,
        
        # 会话类型统计
        'realtime_count': realtime_count,
        'video_count': video_count,
        
        # 疲劳等级统计
        'normal_count': normal_count,
        'mild_count': mild_count,
        'moderate_count': moderate_count,
        'severe_count': severe_count,
        'extreme_count': extreme_count,
        
        # 不良行为统计
        'yawn_count': yawn_count,
        'smoking_count': smoking_count,
        'phone_count': phone_count,
        'drinking_count': drinking_count,
        
        # 近7天趋势
        'week_labels': json.dumps(week_labels),
        'week_data': week_data,
        
        # 时间段分布
        'hourly_data': hourly_data,
    }
    
    return render(request, 'detection/statistics.html', context)

@login_required
def system_config(request):
    """
    系统配置页面，仅管理员可访问
    """
    if not request.user.is_admin and not request.user.is_superuser:
        messages.error(request, '您没有访问此页面的权限')
        return redirect('detection:dashboard')

    from .utils.perf_presets import build_preset, detect_cuda, preset_catalog

    bool_keys = {
        'enable_voice',
        'mono_camera_mode',
        'yolo_spatial_filter',
        'cuda_half',
    }

    cuda_ok, cuda_name = detect_cuda()

    if request.method == 'POST':
        posted = {}
        for key in request.POST.keys():
            if key.startswith('config_'):
                posted[key[7:]] = request.POST.get(key)

        # One-click preset overwrites performance-related keys
        apply_preset = (request.POST.get('apply_preset') or '').strip().lower()
        if apply_preset in ('cpu_smooth', 'balanced', 'gpu_quality'):
            if apply_preset == 'gpu_quality' and not cuda_ok:
                messages.warning(
                    request,
                    '当前机器未检测到可用 NVIDIA GPU，已改为「均衡」预设（CPU）。',
                )
            posted.update(build_preset(apply_preset, cuda_ok))

        for key in bool_keys:
            if key not in posted:
                posted[key] = 'false'

        for config_key, value in posted.items():
            if config_key in bool_keys:
                value = 'true' if str(value).lower() in ('1', 'true', 'on', 'yes') else 'false'
            SystemConfig.objects.update_or_create(
                config_key=config_key,
                defaults={'config_value': value},
            )

        try:
            from .utils.config_cache import invalidate
            invalidate()
        except Exception:  # noqa: BLE001
            pass
        try:
            from .utils.compute_scheduler import compute_scheduler

            compute_scheduler.configure(
                {c.config_key: c.config_value for c in SystemConfig.objects.all()}
            )
            if yolo_detector is not None:
                yolo_detector.load_config()
            if dlib_detector is not None:
                dlib_detector.load_config()
        except Exception:  # noqa: BLE001
            pass

        if apply_preset:
            messages.success(request, '已应用性能预设并保存')
        else:
            messages.success(request, '系统配置已更新')
        return redirect('detection:system_config')

    configs_dict = {}
    for config in SystemConfig.objects.all():
        configs_dict[config.config_key] = config.config_value

    catalog = preset_catalog(cuda_ok)
    return render(
        request,
        'detection/system_config.html',
        {
            'configs': configs_dict,
            'cuda_available': cuda_ok,
            'cuda_name': cuda_name,
            'preset_meta': catalog['presets'],
            'preset_values_json': json.dumps(catalog['values'], ensure_ascii=False),
            'current_preset': configs_dict.get('performance_preset') or 'balanced',
        },
    )

@login_required
def start_detection(request):
    """
    API: 开始检测
    """
    if request.method == 'POST':
        session_id = request.POST.get('session_id')
        source_type = (request.POST.get('source_type') or 'browser').strip().lower()
        
        if not session_id:
            return JsonResponse({'status': 'error', 'message': '缺少会话ID'})
        
        try:
            session = DetectionSession.objects.get(id=session_id, user=request.user)
            configs = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
            from .utils.fatigue_tracker import behavior_tracker, fatigue_tracker

            fatigue_tracker.reset(session.id)
            behavior_tracker.reset(session.id)
            fatigue_tracker.configure(session.id, configs)
            behavior_tracker.configure(session.id, configs)
            try:
                from .utils.compute_scheduler import compute_scheduler

                compute_scheduler.configure(configs)
                if yolo_detector is not None:
                    yolo_detector.load_config()
                if dlib_detector is not None:
                    dlib_detector.load_config()
                # Browser cold-start warmup only. MVS path warms YOLO/dlib inside
                # the existing 5s startup window (alongside exposure calibration).
                if source_type != 'mvs':
                    warm_ms = {}
                    if yolo_detector is not None:
                        warm_ms['yolo'] = round(yolo_detector.warmup(runs=2), 1)
                    if dlib_detector is not None:
                        warm_ms['dlib'] = round(dlib_detector.warmup(), 1)
                    if warm_ms:
                        _safe_log(f"detector warmup ms: {warm_ms}")
            except Exception:  # noqa: BLE001
                pass

            if source_type == 'mvs':
                from .utils.hik_mvs.grabber import mvs_grabber
                interval = int(configs.get('detection_interval', 500))
                camera_ip = (request.POST.get('camera_ip') or '').strip() or None
                device_index = request.POST.get('device_index')
                idx = int(device_index) if device_index not in (None, '') else int(configs.get('default_mvs_index', 0))
                try:
                    if mvs_grabber.running:
                        mvs_grabber.stop(complete_session=False)
                    mvs_grabber.start(
                        session_id=session.id,
                        user_id=request.user.id,
                        interval_ms=interval,
                        device_index=idx if not camera_ip else None,
                        camera_ip=camera_ip,
                    )
                except Exception as exc:  # noqa: BLE001
                    return JsonResponse({'status': 'error', 'message': str(exc)})
            else:
                # Browser / webcam: ensure leftover MVS grabber is released
                try:
                    from .utils.hik_mvs.grabber import mvs_grabber
                    if mvs_grabber.running:
                        mvs_grabber.stop(complete_session=False)
                except Exception:  # noqa: BLE001
                    pass
                # A fresh session must not inherit a stale sticky primary-face
                # bbox left behind by a previous session on this detector
                # singleton (archive path keys it by session id).
                try:
                    if yolo_detector is not None:
                        yolo_detector._sticky_primary_by_session.pop(session.id, None)
                except Exception:  # noqa: BLE001
                    pass
            return JsonResponse({
                'status': 'success',
                'session_id': session.id,
                'source_type': source_type,
            })
        except DetectionSession.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': '无效的会话ID'})
    
    return JsonResponse({'status': 'error', 'message': '不支持的请求方法'})


@login_required
def reset_fatigue(request):
    """API: 用户确认疲劳告警后，清空 PERCLOS/微睡/哈欠累计，避免历史拖累后续判断。"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': '不支持的请求方法'})

    session_id = request.POST.get('session_id')
    if not session_id:
        return JsonResponse({'status': 'error', 'message': '缺少会话ID'})

    try:
        session = DetectionSession.objects.get(id=session_id, user=request.user)
    except DetectionSession.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '无效的会话ID'})

    from .utils.fatigue_tracker import fatigue_tracker

    configs = {c.config_key: c.config_value for c in SystemConfig.objects.all()}
    snap = fatigue_tracker.clear_history(session.id, configs)
    try:
        from .utils.hik_mvs.grabber import mvs_grabber
        with mvs_grabber._lock:
            if mvs_grabber._session_id == session.id:
                mvs_grabber._last_fatigue_level = 0
                mvs_grabber._last_yawn = False
                mvs_grabber._pending_fatigue_event = False
    except Exception:  # noqa: BLE001
        pass
    return JsonResponse({
        'status': 'success',
        'session_id': session.id,
        'fatigue_level': int(snap.fatigue_level),
        'perclos': float(snap.perclos),
        'eye_closed_ms': int(snap.eye_closed_ms),
        'yawn_detected': bool(snap.yawn_detected),
        'is_microsleep': bool(snap.is_microsleep),
    })


@login_required
def stop_detection(request):
    """
    API: 停止检测
    """
    if request.method == 'POST':
        session_id = request.POST.get('session_id')
        
        if not session_id:
            return JsonResponse({'status': 'error', 'message': '缺少会话ID'})
        
        try:
            session = DetectionSession.objects.get(id=session_id, user=request.user)
            try:
                from .utils.hik_mvs.grabber import mvs_grabber
                from .utils.fatigue_tracker import behavior_tracker, fatigue_tracker

                if mvs_grabber.running and mvs_grabber.status().get('session_id') == session.id:
                    mvs_grabber.stop(complete_session=True)
                else:
                    session.end_time = timezone.now()
                    session.status = 'completed'
                    session.save(update_fields=['end_time', 'status'])
                fatigue_tracker.reset(session.id)
                behavior_tracker.reset(session.id)
            except Exception:  # noqa: BLE001
                session.end_time = timezone.now()
                session.status = 'completed'
                session.save(update_fields=['end_time', 'status'])
            return JsonResponse({'status': 'success', 'session_id': session.id})
        except DetectionSession.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': '无效的会话ID'})
    
    return JsonResponse({'status': 'error', 'message': '不支持的请求方法'})


@login_required
def list_sources(request):
    """API: list GigE / industrial cameras from MVS SDK."""
    from .utils.hik_mvs.camera import enumerate_devices
    data = enumerate_devices()
    return JsonResponse({'status': 'success', **data})


@login_required
def mvs_status(request):
    """API: MVS grabber status + latest detection meta."""
    from .utils.hik_mvs.grabber import mvs_grabber
    session_id = request.GET.get('session_id')
    if not session_id:
        return JsonResponse({'status': 'error', 'message': '缺少会话ID'}, status=400)
    try:
        session = DetectionSession.objects.get(id=session_id, user=request.user)
    except DetectionSession.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': '无效的会话ID'}, status=404)
    st = mvs_grabber.status()
    if st.get('session_id') and int(st['session_id']) != int(session.id):
        return JsonResponse({'status': 'error', 'message': '会话与采集器不匹配'}, status=400)
    return JsonResponse({'status': 'success', **st})


@login_required
def mvs_preview(request):
    """MJPEG preview stream for MVS grabber annotated frames."""
    import time
    from django.http import StreamingHttpResponse, HttpResponse
    from .utils.hik_mvs.grabber import mvs_grabber

    session_id = request.GET.get('session_id')
    if not session_id:
        return HttpResponse('missing session_id', status=400)
    try:
        session = DetectionSession.objects.get(id=session_id, user=request.user)
    except DetectionSession.DoesNotExist:
        return HttpResponse('invalid session', status=404)

    def frame_generator():
            idle = 0
            last_jpeg = None
            while True:
                st = mvs_grabber.status()
                if not st.get('running') or st.get('session_id') != session.id:
                    idle += 1
                    if idle > 50:
                        break
                    time.sleep(0.05)
                    continue
                idle = 0
                jpeg = mvs_grabber.get_jpeg()
                # Only push when frame changed to reduce bandwidth; still poll quickly
                if jpeg and jpeg is not last_jpeg:
                    last_jpeg = jpeg
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
                    )
                time.sleep(0.05)

    response = StreamingHttpResponse(
        frame_generator(),
        content_type='multipart/x-mixed-replace; boundary=frame',
    )
    return response

@login_required
def get_result(request):
    """
    API: 接收前端发送的图像帧，进行检测并返回结果
    """
    if request.method == 'POST':
        session_id = request.POST.get('session_id')
        image_data = request.POST.get('image_data')
        detect_fatigue = request.POST.get('detect_fatigue') == 'true'
        detect_behaviors = request.POST.get('detect_behaviors') == 'true'
        
        _safe_log(f"收到检测请求: session_id={session_id}, detect_fatigue={detect_fatigue}, detect_behaviors={detect_behaviors}")
        
        if not session_id or not image_data:
            return JsonResponse({'status': 'error', 'message': '缺少必要参数'})
        
        try:
            session = DetectionSession.objects.get(id=session_id, user=request.user)
            
            # 解码Base64图像
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                return JsonResponse({'status': 'error', 'message': '图像解码失败'})

            from .utils.config_cache import get_configs
            import time as _time

            configs = get_configs()

            # Unified archive detect. The frontend draws the live overlay from the
            # JSON landmarks/detections (drawLiveOverlay) and only falls back to a
            # server-rendered annotated JPEG when landmarks are absent, so only
            # render/encode that (full-res copy + draw + JPEG) in that fallback case
            # instead of doing it synchronously on every request.
            result = process_image_archive(
                image,
                session,
                detect_fatigue,
                detect_behaviors,
                include_image_data=False,
            )
            if not (result.get('landmarks') and len(result['landmarks'])):
                from .utils.archive_pipeline import encode_jpeg_bgr, render_annotated_bgr

                annotated = render_annotated_bgr(image, result)
                raw = encode_jpeg_bgr(annotated, 75)
                if raw is not None:
                    result['image_data'] = (
                        f"data:image/jpeg;base64,{base64.b64encode(raw).decode('utf-8')}"
                    )

            try:
                persist_iv = float(configs.get('yolo_persist_interval_sec') or 2.0)
            except (TypeError, ValueError):
                persist_iv = 2.0
            persist_iv = max(0.5, min(30.0, persist_iv))
            sid = int(session.id)
            now = _time.time()
            last_at = float(_browser_last_persist_at.get(sid) or 0.0)
            due = (now - last_at) >= persist_iv
            behavior_hit = bool(
                result.get('smoking_detected')
                or result.get('phone_detected')
                or result.get('drinking_detected')
            )
            level = int(result.get('fatigue_level') or 0)
            yawn = bool(result.get('yawn_detected'))
            prev_level = int(_browser_last_fatigue_level.get(sid) or 0)
            prev_yawn = bool(_browser_last_yawn.get(sid))
            fatigue_edge = (level > prev_level and level >= 2) or ((not prev_yawn) and yawn)
            _browser_last_fatigue_level[sid] = level
            _browser_last_yawn[sid] = yawn

            if due or behavior_hit or fatigue_edge:
                result = persist_detection_snapshot(image, session, result)
                _browser_last_persist_at[sid] = now

            try:
                from .utils.persist_worker import stats as _persist_stats

                result["persist"] = _persist_stats()
            except Exception:  # noqa: BLE001
                pass

            # JsonResponse cannot encode numpy / private buffers
            for k in list(result.keys()):
                if str(k).startswith("_") or k in (
                    "overlay_landmarks",
                    "overlay_landmarks_size",
                ):
                    result.pop(k, None)
            return JsonResponse(result)
        except Exception as e:
            _safe_log(f"检测过程出现错误: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': '不支持的请求方法'})


def persist_detection_snapshot(image, session, results):
    """Queue JPEG encode + DetectionResult write on a background worker so the
    hot request/detect-loop thread never blocks on disk/DB latency (matters a
    lot for high-resolution GigE frames). `detection_id` / `result_image_url`
    are not known synchronously anymore; neither is rendered by the frontend,
    so this is a pure perf win with no visible behavior change."""
    from .utils.persist_worker import enqueue_persist

    enqueue_persist(image, session, results)
    results["detection_id"] = None
    results["result_image_url"] = None
    return results


def process_image_archive(
    image,
    session,
    detect_fatigue=True,
    detect_behaviors=True,
    include_image_data=False,
):
    """
    Unified archive path for browser + MVS.
    Uniform max-side scale (FOV kept) so HOG sees archive-like face sizes;
    geometry remapped to original for live overlay. JPEG only when requested.
    """
    from .utils.archive_pipeline import (
        encode_jpeg_bgr,
        render_annotated_bgr,
        run_archive_detect,
    )
    from .utils.config_cache import get_configs

    configs = get_configs()
    try:
        max_side = int(float(configs.get("yolo_detect_max_width", 960) or 960))
    except (TypeError, ValueError):
        max_side = 960
    if max_side < 0:
        max_side = 0
    dlib_refine_mode = str(configs.get("dlib_refine_mode") or "off").strip().lower()
    if dlib_refine_mode not in ("off", "light", "full"):
        dlib_refine_mode = "off"
    dlib_landmark_mode = str(configs.get("dlib_landmark_mode") or "yolo").strip().lower()

    sid = getattr(session, "id", None)
    results = run_archive_detect(
        image,
        sid,
        detect_fatigue=detect_fatigue,
        detect_behaviors=detect_behaviors,
        max_side=max_side,
        dlib_refine_mode=dlib_refine_mode,
        dlib_landmark_mode=dlib_landmark_mode,
    )
    # Drop any non-JSON leftovers
    for k in list(results.keys()):
        if str(k).startswith("_"):
            results.pop(k, None)
    results["image_data"] = None
    if include_image_data:
        annotated = render_annotated_bgr(image, results)
        raw = encode_jpeg_bgr(annotated, 75)
        if raw is not None:
            results["image_data"] = (
                f"data:image/jpeg;base64,{base64.b64encode(raw).decode('utf-8')}"
            )
    return results


def process_image(
    image,
    session,
    detect_fatigue=True,
    detect_behaviors=True,
    include_image_data=True,
    persist=True,
):
    """
    处理图像，进行检测并保存结果
    :param persist: False 时只推理不画图/写库（MVS 高频路径，显著降低延迟）
    """
    import time as _time

    from .utils.fatigue_tracker import behavior_tracker, fatigue_tracker
    from .utils.mono_preprocess import enhance_for_mono, load_mono_config

    t0 = _time.perf_counter()
    timing = {
        'mono_ms': 0.0,
        'yolo_ms': 0.0,
        'dlib_ms': 0.0,
        'draw_ms': 0.0,
        'save_ms': 0.0,
        'total_ms': 0.0,
    }
    results = {'status': 'success', 'face_bbox': None}
    from .utils.config_cache import get_configs
    configs = get_configs()
    mono_cfg = load_mono_config(configs)

    # Align with MVS / browser: shrink before YOLO when still oversized
    image = _resize_for_detect(image, configs)

    face_bbox = None
    processed_yolo = None
    if detect_behaviors and yolo_detector:
        t_mono = _time.perf_counter()
        if mono_cfg['enabled']:
            yolo_input = enhance_for_mono(image, configs)
            timing['mono_ms'] = round((_time.perf_counter() - t_mono) * 1000.0, 1)
        else:
            yolo_input = image
            timing['mono_ms'] = 0.0

        t_yolo = _time.perf_counter()
        yolo_results = yolo_detector.detect(yolo_input)
        processed_yolo = yolo_detector.process_results(yolo_results)
        timing['yolo_ms'] = round((_time.perf_counter() - t_yolo) * 1000.0, 1)

        confirmed = behavior_tracker.update(
            session.id,
            smoking=bool(processed_yolo['smoking_detected']),
            phone=bool(processed_yolo['phone_detected']),
            drinking=bool(processed_yolo['drinking_detected']),
            configs=configs,
        )

        # Prefer YOLO's primary face (largest); fall back to first face box.
        face_bbox = processed_yolo.get('face_bbox')
        if face_bbox is None:
            for det in processed_yolo.get('detections', []):
                if det.get('class_name') == 'face':
                    face_bbox = det.get('bbox')
                    break
        face_bboxes = processed_yolo.get('face_bboxes') or []
        if not face_bboxes and face_bbox is not None:
            face_bboxes = [face_bbox]

        results.update({
            'face_detected': bool(processed_yolo['face_detected']),
            'smoking_detected': bool(confirmed['smoking_detected']),
            'phone_detected': bool(confirmed['phone_detected']),
            'drinking_detected': bool(confirmed['drinking_detected']),
            'face_bbox': face_bbox,
            'face_bboxes': face_bboxes,
            'face_count': int(processed_yolo.get('face_count') or len(face_bboxes)),
            'behavior_debug': processed_yolo.get('behavior_debug') or {},
            'confirm_progress': confirmed.get('confirm_progress') or {},
            'detections': processed_yolo.get('detections') or [],
        })
    else:
        face_bboxes = []
        results.update({
            'face_detected': False,
            'smoking_detected': False,
            'phone_detected': False,
            'drinking_detected': False,
            'face_bbox': None,
            'face_bboxes': [],
            'face_count': 0,
            'behavior_debug': {},
            'confirm_progress': {},
            'detections': [],
        })

    draw_lm = None
    draw_lm_size = None
    draw_mar = None
    draw_faces = None
    faces_n = 0
    if detect_fatigue and dlib_detector:
        try:
            t_dlib = _time.perf_counter()
            # Default: YOLO-box guided landmarks (fast, tracks motion). HOG only if no face.
            landmark_mode = str(configs.get('dlib_landmark_mode') or 'yolo').strip().lower()
            refine = str(configs.get('dlib_refine_mode') or 'light').strip().lower()
            if refine not in ('off', 'light', 'full'):
                refine = 'light'
            if landmark_mode == 'hog' or not (face_bboxes or face_bbox):
                dlib_results = dlib_detector.detect_fatigue_multi(
                    image,
                    face_bboxes=None,
                    primary_bbox=None,
                    allow_hog=True,
                    refine_rect='off',
                )
            else:
                dlib_results = dlib_detector.detect_fatigue_multi(
                    image,
                    face_bboxes=face_bboxes or ([face_bbox] if face_bbox else None),
                    primary_bbox=face_bbox,
                    allow_hog=True,
                    refine_rect=refine,
                )
            timing['dlib_ms'] = round((_time.perf_counter() - t_dlib) * 1000.0, 1)

            faces_n = int(dlib_results.get('faces_detected') or 0)
            if faces_n <= 0 and results.get('face_detected') and face_bbox:
                faces_n = 1

            # Archive did not gate EAR on pose/quality
            snap = fatigue_tracker.update(
                session.id,
                ear=dlib_results.get('eye_aspect_ratio'),
                yawn_detected=bool(dlib_results.get('yawn_detected')),
                faces_detected=faces_n,
                landmarks=dlib_results.get('landmarks'),
                landmarks_size=(
                    (int(image.shape[1]), int(image.shape[0]))
                    if dlib_results.get('landmarks') is not None
                    else None
                ),
                faces=dlib_results.get('faces') or [],
                landmark_reliable=True,
            )
            draw_mar = dlib_results.get('mouth_aspect_ratio')
            draw_lm = dlib_results.get('landmarks') or snap.landmarks
            draw_faces = dlib_results.get('faces') or getattr(snap, 'faces', None)
            draw_lm_size = (
                (int(image.shape[1]), int(image.shape[0]))
                if dlib_results.get('landmarks') is not None
                else getattr(snap, 'landmarks_size', None)
            )
        except Exception as e:
            _safe_log(f'dlib检测失败(沿用tracker快照): {e}')
            snap = fatigue_tracker.get_snapshot(session.id)
            draw_lm = snap.landmarks
            draw_lm_size = getattr(snap, 'landmarks_size', None)
            draw_faces = getattr(snap, 'faces', None)
            faces_n = int(snap.faces_detected or 0)
            timing['dlib_ms'] = round(timing.get('dlib_ms') or 0.0, 1)

        results.update({
            'eye_aspect_ratio': float(snap.eye_aspect_ratio) if snap.eye_aspect_ratio is not None else None,
            'mouth_aspect_ratio': float(draw_mar) if draw_mar is not None else None,
            'yawn_detected': bool(snap.yawn_detected),
            'fatigue_level': int(snap.fatigue_level),
            'perclos': float(snap.perclos),
            'eye_closed_ms': int(snap.eye_closed_ms),
            'is_microsleep': bool(snap.is_microsleep),
            'has_landmarks': draw_lm is not None,
            'faces': [
                {
                    'ear': f.get('eye_aspect_ratio'),
                    'mar': f.get('mouth_aspect_ratio'),
                    'yawn': bool(f.get('yawn_detected')),
                    'is_primary': bool(f.get('is_primary')),
                    'bbox': f.get('bbox'),
                }
                for f in (draw_faces or [])
            ],
            'face_count': max(int(results.get('face_count') or 0), int(faces_n or 0)),
        })
    else:
        snap = fatigue_tracker.get_snapshot(session.id)
        draw_lm = snap.landmarks
        draw_lm_size = getattr(snap, 'landmarks_size', None)
        draw_faces = getattr(snap, 'faces', None)
        results.update({
            'eye_aspect_ratio': float(snap.eye_aspect_ratio) if snap.eye_aspect_ratio is not None else None,
            'mouth_aspect_ratio': None,
            'yawn_detected': bool(snap.yawn_detected),
            'fatigue_level': int(snap.fatigue_level),
            'perclos': float(snap.perclos),
            'eye_closed_ms': int(snap.eye_closed_ms),
            'is_microsleep': bool(snap.is_microsleep),
            'has_landmarks': snap.landmarks is not None,
            'faces': [
                {
                    'ear': f.get('eye_aspect_ratio'),
                    'mar': f.get('mouth_aspect_ratio'),
                    'yawn': bool(f.get('yawn_detected')),
                    'is_primary': bool(f.get('is_primary')),
                    'bbox': f.get('bbox'),
                }
                for f in (draw_faces or [])
            ],
            'face_count': int(results.get('face_count') or snap.faces_detected or 0),
        })

    if not persist and not include_image_data:
        timing['total_ms'] = round((_time.perf_counter() - t0) * 1000.0, 1)
        results['timing'] = timing
        results['result_image_url'] = None
        results['detection_id'] = None
        results['image_data'] = None
        # Geometry for live browser overlay (no JPEG slideshow)
        results['image_size'] = [int(image.shape[1]), int(image.shape[0])]
        if draw_lm is not None:
            try:
                results['landmarks'] = [
                    [int(p[0]), int(p[1])] for p in np.asarray(draw_lm)
                ]
            except Exception:  # noqa: BLE001
                results['landmarks'] = None
        else:
            results['landmarks'] = None
        if draw_lm_size is not None:
            try:
                results['landmarks_size'] = [int(draw_lm_size[0]), int(draw_lm_size[1])]
            except (TypeError, ValueError, IndexError):
                results['landmarks_size'] = results['image_size']
        else:
            results['landmarks_size'] = results['image_size']
        return results

    t_draw = _time.perf_counter()
    if processed_yolo is not None and yolo_detector:
        image_with_yolo = yolo_detector.draw_results(image, processed_yolo)
    else:
        image_with_yolo = image

    if dlib_detector and (
        draw_lm is not None
        or results.get('eye_aspect_ratio') is not None
        or int(results.get('fatigue_level') or 0) > 0
    ):
        draw_payload = {
            'landmarks': draw_lm,
            'landmarks_size': draw_lm_size,
            'faces': draw_faces,
            'eye_aspect_ratio': results.get('eye_aspect_ratio'),
            'mouth_aspect_ratio': draw_mar,
            'yawn_detected': results.get('yawn_detected'),
            'fatigue_level': results.get('fatigue_level'),
            'perclos': results.get('perclos'),
            'eye_closed_ms': results.get('eye_closed_ms'),
        }
        image_with_results = dlib_detector.draw_fatigue_results(image_with_yolo, draw_payload)
    else:
        image_with_results = image_with_yolo
    timing['draw_ms'] = round((_time.perf_counter() - t_draw) * 1000.0, 1)

    if not persist:
        timing['save_ms'] = 0.0
        timing['total_ms'] = round((_time.perf_counter() - t0) * 1000.0, 1)
        results['timing'] = timing
        results['result_image_url'] = None
        results['detection_id'] = None
        if include_image_data:
            ok, buffer = cv2.imencode('.jpg', image_with_results, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                results['image_data'] = f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"
            else:
                results['image_data'] = None
        else:
            results['image_data'] = None
        return results

    t_save = _time.perf_counter()
    detection_result = DetectionResult(
        session=session,
        face_detected=results['face_detected'],
        smoking_detected=results['smoking_detected'],
        phone_detected=results['phone_detected'],
        drinking_detected=results['drinking_detected'],
        eye_aspect_ratio=results['eye_aspect_ratio'],
        yawn_detected=results['yawn_detected'],
        fatigue_level=results['fatigue_level'],
        perclos=results.get('perclos'),
        eye_closed_ms=results.get('eye_closed_ms'),
    )

    result_filename = f"result_{session.id}_{timezone.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(3).hex()}.jpg"
    try:
        ok, buffer = cv2.imencode('.jpg', image_with_results, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            detection_result.result_image.save(result_filename, ContentFile(buffer.tobytes()), save=False)
        else:
            _safe_log('结果图像 JPEG 编码失败')
    except Exception as e:
        _safe_log(f'保存结果图像失败: {e}')

    detection_result.save()
    timing['save_ms'] = round((_time.perf_counter() - t_save) * 1000.0, 1)
    timing['total_ms'] = round((_time.perf_counter() - t0) * 1000.0, 1)
    results['timing'] = timing
    results['result_image_url'] = detection_result.result_image.url if detection_result.result_image else None
    results['detection_id'] = detection_result.id

    if include_image_data:
        ok, buffer = cv2.imencode('.jpg', image_with_results, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            results['image_data'] = f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"
        else:
            results['image_data'] = None
    else:
        results['image_data'] = None

    return results


@login_required
def history(request):
    """
    检测历史记录页面，以小红书卡片风格展示
    """
    # 获取用户的所有检测会话
    sessions = DetectionSession.objects.filter(user=request.user).order_by('-start_time')
    
    # 获取查询参数
    session_type = request.GET.get('type', '')
    status = request.GET.get('status', '')
    
    # 根据筛选条件过滤会话
    if session_type:
        sessions = sessions.filter(session_type=session_type)
    if status:
        sessions = sessions.filter(status=status)
    
    # 为每个会话获取代表性图片和摘要数据
    for session in sessions:
        # 获取该会话的一张有代表性的图片（如果有）
        representative_result = DetectionResult.objects.filter(
            session=session, 
            result_image__isnull=False
        ).first()
        
        session.thumbnail = representative_result.result_image if representative_result else None
        
        # 获取会话的摘要数据
        session.detection_count = session.get_detection_count()
        session.fatigue_count = DetectionResult.objects.filter(
            session=session, 
            fatigue_level__gte=2
        ).count()
        session.behavior_count = DetectionResult.objects.filter(
            session=session
        ).filter(
            smoking_detected=True
        ).count() + DetectionResult.objects.filter(
            session=session
        ).filter(
            phone_detected=True
        ).count()
    
    context = {
        'sessions': sessions,
        'session_types': dict(DetectionSession.SESSION_TYPES),
        'session_statuses': dict(DetectionSession.SESSION_STATUS),
        'current_type': session_type,
        'current_status': status,
    }
    
    return render(request, 'detection/history.html', context)
