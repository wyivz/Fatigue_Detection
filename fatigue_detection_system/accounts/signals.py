from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth import get_user_model

User = get_user_model()

@receiver(post_save, sender=User)
def save_profile(sender, instance, created, **kwargs):
    """原本应该保存用户配置文件，但为了避免错误，我们不做任何操作"""
    pass 