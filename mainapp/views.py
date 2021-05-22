from django.http.response import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login as auth_login

from .forms import UserCreationForm, AuthenticationForm

import base64
import cv2
import numpy as np

@login_required
def index(req):
    return render(req, 'index/index.html')

def analyze(req):
    print(req.GET)
    print(req.POST)
    print(req.FILES)
    if req.FILES:
        img_res = {}
        img_px = []
        for file in req.FILES.values():
            img=cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0,70,70])
            upper_red = np.array([20,200,150])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
            lower_red = np.array([170,70,70])
            upper_red = np.array([180,200,150])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
            mask = mask0+mask1
            img_px.append(np.sum(mask)/255)
            al = cv2.bitwise_and(img,img,mask=mask)
            dst = cv2.addWeighted(img,0.1,al,0.9,0)
            imencoded = cv2.imencode("hello.jpg", dst)[1]
            # response_pickled  = jsonpickle.encode(imencoded)
            # filename="res"+str(image_file.split("/")[-1].split(".")[0])+'.jpg'
            # file_name['filename'+str(i)]=filename
            # file_name['filename'+str(i+5)]=str(image_file.split("/")[-1].split(".")[0])+'.jpg'
            # print(imencoded.tostring())
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'",'')
            
            with open('newfile.txt', 'w') as file:
                file.write(str(img_res))
    img_res['px_details'] = img_px
    return JsonResponse(img_res)
    # return HttpResponse(imencoded.tostring(), content_type="image/png")

def signup_view(request):
    form = UserCreationForm()
    if request.POST:
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    return render(request, 'index/register.html', {'form': form})