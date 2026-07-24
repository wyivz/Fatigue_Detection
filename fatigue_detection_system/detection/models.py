from django.db import models
from django.conf import settings

class DetectionSession(models.Model):
    """
    检测会话模型，记录每次检测的基本信息
    """
    SESSION_TYPES = (
        ('realtime', '实时检测'),
        ('video', '视频检测')
    )
    
    SESSION_STATUS = (
        ('in_progress', '进行中'),
        ('completed', '已完成'),
        ('failed', '失败')
    )
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, 
                             related_name='detection_sessions', verbose_name="用户")
    start_time = models.DateTimeField(auto_now_add=True, verbose_name="开始时间")
    end_time = models.DateTimeField(null=True, blank=True, verbose_name="结束时间")
    session_type = models.CharField(max_length=20, choices=SESSION_TYPES, verbose_name="检测类型")
    source_file = models.FileField(upload_to='uploads/', null=True, blank=True, verbose_name="源文件")
    status = models.CharField(max_length=20, choices=SESSION_STATUS, default='in_progress', verbose_name="状态")
    
    class Meta:
        verbose_name = "检测会话"
        verbose_name_plural = verbose_name
        ordering = ['-start_time']
    
    def __str__(self):
        return f"{self.user.username}的{self.get_session_type_display()}({self.start_time})"
        
    def get_detection_count(self):
        """获取当前会话的检测结果数量"""
        return self.detection_results.count()
        
    def get_latest_result(self):
        """获取最新的检测结果"""
        return self.detection_results.order_by('-timestamp').first()
        
    @property
    def normal_percentage(self):
        """计算正常状态的百分比"""
        total = self.detection_results.count()
        if total == 0:
            return 0
        normal_count = self.detection_results.filter(fatigue_level=0).count()
        return round((normal_count / total) * 100)
    
    @property
    def mild_fatigue_percentage(self):
        """计算轻度疲劳的百分比"""
        total = self.detection_results.count()
        if total == 0:
            return 0
        mild_count = self.detection_results.filter(fatigue_level=1).count()
        return round((mild_count / total) * 100)
    
    @property
    def moderate_fatigue_percentage(self):
        """计算中度疲劳的百分比"""
        total = self.detection_results.count()
        if total == 0:
            return 0
        moderate_count = self.detection_results.filter(fatigue_level=2).count()
        return round((moderate_count / total) * 100)
    
    @property
    def severe_fatigue_percentage(self):
        """计算较严重疲劳的百分比"""
        total = self.detection_results.count()
        if total == 0:
            return 0
        severe_count = self.detection_results.filter(fatigue_level=3).count()
        return round((severe_count / total) * 100)
    
    @property
    def extreme_fatigue_percentage(self):
        """计算严重疲劳的百分比"""
        total = self.detection_results.count()
        if total == 0:
            return 0
        extreme_count = self.detection_results.filter(fatigue_level=4).count()
        return round((extreme_count / total) * 100)
    
    @property
    def yawn_count(self):
        """打哈欠次数"""
        return self.detection_results.filter(yawn_detected=True).count()
    
    @property
    def smoking_count(self):
        """抽烟次数"""
        return self.detection_results.filter(smoking_detected=True).count()
    
    @property
    def phone_count(self):
        """打电话次数"""
        return self.detection_results.filter(phone_detected=True).count()
    
    @property
    def drinking_count(self):
        """喝水次数"""
        return self.detection_results.filter(drinking_detected=True).count()

class DetectionResult(models.Model):
    """
    检测结果模型，记录每帧的检测结果
    """
    session = models.ForeignKey(DetectionSession, on_delete=models.CASCADE, 
                               related_name='detection_results', verbose_name="检测会话")
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name="时间戳")
    
    # YOLO检测结果
    face_detected = models.BooleanField(default=False, verbose_name="检测到面部")
    smoking_detected = models.BooleanField(default=False, verbose_name="检测到抽烟")
    phone_detected = models.BooleanField(default=False, verbose_name="检测到打电话")
    drinking_detected = models.BooleanField(default=False, verbose_name="检测到喝水")
    
    # dlib检测结果
    eye_aspect_ratio = models.FloatField(null=True, blank=True, verbose_name="眼睛纵横比")
    yawn_detected = models.BooleanField(default=False, verbose_name="检测到打哈欠")
    
    # 综合判断
    fatigue_level = models.IntegerField(default=0, verbose_name="疲劳等级")  # 0-4 疲劳程度
    
    result_image = models.ImageField(upload_to='results/', null=True, blank=True, verbose_name="结果图像")
    
    class Meta:
        verbose_name = "检测结果"
        verbose_name_plural = verbose_name
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"检测结果 {self.id} - {self.timestamp}"
        
    @property
    def is_fatigued(self):
        """判断是否疲劳"""
        return self.fatigue_level >= 2
        
    @property
    def has_unsafe_behavior(self):
        """判断是否有不安全行为"""
        return self.smoking_detected or self.phone_detected

class SystemConfig(models.Model):
    """
    系统配置模型，存储系统的配置参数
    """
    config_key = models.CharField(max_length=50, unique=True, verbose_name="配置键")
    config_value = models.CharField(max_length=255, verbose_name="配置值")
    description = models.TextField(blank=True, null=True, verbose_name="描述")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    
    class Meta:
        verbose_name = "系统配置"
        verbose_name_plural = verbose_name
    
    def __str__(self):
        return self.config_key
