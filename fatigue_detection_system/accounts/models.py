from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    """
    自定义用户模型，扩展 Django 内置的 User 模型
    """
    is_admin = models.BooleanField(default=False, verbose_name="是否为管理员")
    phone = models.CharField(max_length=20, blank=True, null=True, verbose_name="手机号码")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    
    class Meta:
        verbose_name = "用户"
        verbose_name_plural = verbose_name
    
    def __str__(self):
        return self.username
