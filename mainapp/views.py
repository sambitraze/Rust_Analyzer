import os
from django.http.response import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login as auth_login

from .forms import UserCreationForm, AuthenticationForm

import base64
import cv2
import numpy as np
import tensorflow as tf
from google.cloud import vision


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'creds.json'
client = vision.ImageAnnotatorClient()
classes = ["background","number plate"]
np.set_printoptions(suppress=True)

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5


@login_required
def index(req):    
    dashboard = "rust"
    context = {'dashboard': dashboard}
    return render(req, 'index/index.html',context)

def indexrust(req):
    dashboard = "rust"
    context = {'dashboard': dashboard}
    return render(req, 'index/indexrust.html',context)

def indexanpr(req):
    dashboard = "anpr"
    context = {'dashboard': dashboard}
    return render(req, 'index/indexanpr.html',context)

def indexscratch(req):
    dashboard = "scratch"
    context = {'dashboard': dashboard}
    return render(req, 'index/indexscratch.html',context)

def analyzerust(req):
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
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'",'')
            
            with open('newfile.txt', 'w') as file:
                file.write(str(img_res))
    img_res['px_details'] = img_px
    return JsonResponse(img_res)

def analyzeanpr(req):
    if req.FILES:
        img_res = {}
        img_px = []
        for file in req.FILES.values():
            img=cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            colors = np.random.uniform(0,255,size=(len(classes),3))
            with tf.io.gfile.GFile('num_plate.pb','rb') as f:
        	    graph_def=tf.compat.v1.GraphDef()
        	    graph_def.ParseFromString(f.read())
            with tf.compat.v1.Session() as sess:
    	        sess.graph.as_default()
    	        tf.import_graph_def(graph_def, name='')
    	        rows=img.shape[0]
    	        cols=img.shape[1]
    	        inp=cv2.resize(img,(220,220))
    	        inp=inp[:,:,[2,1,0]]
    	        out=sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
						                                sess.graph.get_tensor_by_name('detection_scores:0'),
                      		            sess.graph.get_tensor_by_name('detection_boxes:0'),
                      		            sess.graph.get_tensor_by_name('detection_classes:0')],
                     		            feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1],3)})
    	        num_detections=int(out[0][0])
    	        for i in range(num_detections):
    		        classId = int(out[3][0][i])
    		        score=float(out[1][0][i])
    		        bbox=[float(v) for v in out[2][0][i]]
    		        label=classes[classId]
    		        if (score>0.3):
    			        x=bbox[1]*cols
    			        y=bbox[0]*rows
    			        right=bbox[3]*cols
    			        bottom=bbox[2]*rows
    			        color=colors[classId]
    			        cv2.rectangle(img, (int(x), int(y)), (int(right),int(bottom)), color, thickness=1)
    			        crop = img[int(y):int(bottom), int(x):int(right)]
    			        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    			        Cropped = cv2.resize(gray,(300,100))
    			        ret, thresh4 = cv2.threshold(Cropped, 120, 255, cv2.THRESH_TOZERO) 
    			        success, encoded_image = cv2.imencode('.png', thresh4)
    			        data = encoded_image.tobytes()
    			        print(type(data))
    			        image = vision.Image(content=data)
    			        response = client.document_text_detection(image=image)
    			        doc = response.full_text_annotation.text
    img_res['px_details'] = doc
    return JsonResponse(img_res)

def analyzescratch(req):
    class_names = []
    with open("coco.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    if req.FILES:
        img_res = {}
        img_px = []
        for file in req.FILES.values():
            img=cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            colors = np.random.uniform(0,255,size=(len(class_names),3))
            net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            model = cv2.dnn_DetectionModel(net)
            model.setInputParams(size=(640, 640), scale=1/255, swapRB=True) 
            classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            for (classid, score, box) in zip(classes, scores, boxes):
                color=colors[classid[0]]
                label = "%s : %f" % (class_names[classid[0]], score)
                cv2.rectangle(img, box,color, 2)
                cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
            imencoded = cv2.imencode("hello.jpg", img)[1]            
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'",'')            
            with open('newfile.txt', 'w') as file:
                file.write(str(img_res))
    img_res['px_details'] = img_px
    return JsonResponse(img_res)

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
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'",'')
            
            with open('newfile.txt', 'w') as file:
                file.write(str(img_res))
    img_res['px_details'] = img_px
    return JsonResponse(img_res)

def signup_view(request):
    form = UserCreationForm()
    if request.POST:
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    return render(request, 'index/register.html', {'form': form})