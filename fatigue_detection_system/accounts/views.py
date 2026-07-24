from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import User
from django import forms

class LoginForm(forms.Form):
    """
    用户登录表单
    """
    username = forms.CharField(
        label='用户名',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入用户名'})
    )
    password = forms.CharField(
        label='密码',
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请输入密码'})
    )

class RegisterForm(forms.Form):
    """
    用户注册表单
    """
    username = forms.CharField(
        label='用户名',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入用户名'})
    )
    email = forms.EmailField(
        label='邮箱',
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': '请输入邮箱'})
    )
    password = forms.CharField(
        label='密码',
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请输入密码'})
    )
    password2 = forms.CharField(
        label='确认密码',
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '请再次输入密码'})
    )
    phone = forms.CharField(
        label='手机号码',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '请输入手机号码（选填）'})
    )
    
    def clean_password2(self):
        """验证两次输入的密码是否一致"""
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError('两次输入的密码不一致')
        return cd['password2']
    
    def clean_username(self):
        """验证用户名是否已经存在"""
        username = self.cleaned_data['username']
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError('用户名已存在')
        return username

class ProfileForm(forms.ModelForm):
    """
    个人资料编辑表单
    """
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'phone']
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control'})
        }

def user_login(request):
    """
    用户登录视图
    """
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            user = authenticate(request, username=cd['username'], password=cd['password'])
            if user is not None:
                if user.is_active:
                    login(request, user)
                    # 获取登录后重定向的URL，默认为仪表盘
                    next_url = request.GET.get('next', '/detection/dashboard/')
                    return redirect(next_url)
                else:
                    messages.error(request, '账户已禁用')
            else:
                messages.error(request, '用户名或密码错误')
    else:
        form = LoginForm()
    
    return render(request, 'accounts/login.html', {'form': form})

def user_register(request):
    """
    用户注册视图
    """
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            # 创建新用户
            user = User.objects.create_user(
                username=cd['username'],
                email=cd['email'],
                password=cd['password']
            )
            # 设置额外信息
            if cd.get('phone'):
                user.phone = cd['phone']
            user.save()
            
            # 登录新用户
            login(request, user)
            messages.success(request, '注册成功！欢迎使用疲劳驾驶检测系统')
            return redirect('/detection/dashboard/')
    else:
        form = RegisterForm()
    
    return render(request, 'accounts/register.html', {'form': form})

@login_required
def user_profile(request):
    """
    用户个人资料视图
    """
    return render(request, 'accounts/profile.html', {'user': request.user})

@login_required
def update_profile(request):
    """
    更新个人资料视图
    """
    if request.method == 'POST':
        # 获取表单数据
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        
        # 更新用户信息
        user = request.user
        user.email = email
        user.first_name = first_name
        user.save()
        
        messages.success(request, '个人资料更新成功')
        return redirect('/accounts/profile/')
    
    return redirect('/accounts/profile/')

@login_required
def edit_profile(request):
    """
    编辑个人资料视图
    """
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, '个人资料更新成功')
            return redirect('/accounts/profile/')
    else:
        form = ProfileForm(instance=request.user)
    
    return render(request, 'accounts/edit_profile.html', {'form': form})
