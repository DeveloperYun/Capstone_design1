from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from accounts.forms import UserForm
from django.contrib.auth.decorators import login_required


def signup(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)  # 사용자 인증
            login(request, user)  # 로그인
            return redirect('index')
    else:
        form = UserForm()
    return render(request, 'accounts/signup.html', {'form': form})

@login_required
def profile_edit(request):
    return render(request, "accounts/profile_edit_form.html",{
        
    })