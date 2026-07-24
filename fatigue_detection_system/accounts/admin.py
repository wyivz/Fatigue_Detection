from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'is_admin', 'is_active')
    list_filter = ('is_staff', 'is_admin', 'is_active')
    fieldsets = UserAdmin.fieldsets + (
        ('额外信息', {'fields': ('is_admin', 'phone', 'created_at')}),
    )
    readonly_fields = ('created_at',)
