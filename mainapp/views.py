import os
import glob
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
import easyocr
from imageio import imread, imsave

reader = easyocr.Reader(['en'])

import PIL
from PIL import ImageDraw
from PIL import Image as imm

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'creds.json'
client = vision.ImageAnnotatorClient()
classes = ["background", "number plate"]
np.set_printoptions(suppress=True)

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5


@login_required
def index(req):
    dashboard = "rust"
    context = {'dashboard': dashboard}
    return render(req, 'index/index.html', context)


def indexrust(req):
    dashboard = "rust"
    context = {'dashboard': dashboard}
    return render(req, 'index/indexrust.html', context)


def indexanpr(req):
    dashboard = "anpr"
    context = {'dashboard': dashboard}
    return render(req, 'index/indexanpr.html', context)


def indexscratch(req):
    dashboard = "scratch"
    context = {'dashboard': dashboard}
    return render(req, 'index/indexscratch.html', context)


def indexiocr(req):
    dashboard = "iocr"
    context = {'dashboard': dashboard}
    return render(req, 'index/indexiocr.html', context)


def indexobjectcount(req):
    dashboard = "objectcount"
    context = {'dashboard': dashboard}
    return render(req, 'index/indexobjectcount.html', context)


def indexfacebeauty(req):
    dashboard = "facebeauty"
    context = {'dashboard': dashboard}
    return render(req, 'index/indexfacebeauty.html', context)


def analyzerust(req):
    if req.FILES:
        img_res = {}
        img_px = []
        for file in req.FILES.values():
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 70, 70])
            upper_red = np.array([20, 200, 150])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
            lower_red = np.array([170, 70, 70])
            upper_red = np.array([180, 200, 150])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
            mask = mask0 + mask1
            img_px.append(np.sum(mask) / 255)
            al = cv2.bitwise_and(img, img, mask=mask)
            dst = cv2.addWeighted(img, 0.1, al, 0.9, 0)
            imencoded = cv2.imencode("hello.jpg", dst)[1]
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'", '')
            with open('newfile.txt', 'w') as file:
                file.write(str(img_res))
    img_res['px_details'] = img_px
    return JsonResponse(img_res)


def analyzeanpr(req):
    if req.FILES:
        img_res = {}
        img_px = []
        for file in req.FILES.values():
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            with tf.io.gfile.GFile('num_plate.pb', 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.compat.v1.Session() as sess:
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                rows = img.shape[0]
                cols = img.shape[1]
                inp = cv2.resize(img, (220, 220))
                inp = inp[:, :, [2, 1, 0]]
                out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                sess.graph.get_tensor_by_name('detection_classes:0')],
                               feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
                num_detections = int(out[0][0])
                for i in range(num_detections):
                    classId = int(out[3][0][i])
                    score = float(out[1][0][i])
                    bbox = [float(v) for v in out[2][0][i]]
                    label = classes[classId]
                    if (score > 0.3):
                        x = bbox[1] * cols
                        y = bbox[0] * rows
                        right = bbox[3] * cols
                        bottom = bbox[2] * rows
                        color = colors[classId]
                        cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), color, thickness=1)
                        crop = img[int(y):int(bottom), int(x):int(right)]
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        Cropped = cv2.resize(gray, (300, 100))
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
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            colors = np.random.uniform(0, 255, size=(len(class_names), 3))
            net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            model = cv2.dnn_DetectionModel(net)
            model.setInputParams(size=(640, 640), scale=1 / 255, swapRB=True)
            classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            for (classid, score, box) in zip(classes, scores, boxes):
                color = colors[classid[0]]
                label = "%s : %f" % (class_names[classid[0]], score)
                # cv2.rectangle(img, box,color, 2)
                box[0] = box[0] - 50
                box[1] = box[1] - 50
                box[2] = box[2] + 50
                box[3] = box[3] + 50
                cv2.rectangle(img, box, [0, 0, 255], 5)
                cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            imencoded = cv2.imencode("hello.jpg", img)[1]
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'", '')
            with open('newfile.txt', 'w') as file:
                file.write(str(img_res))
    img_res['px_details'] = img_px
    return JsonResponse(img_res)


def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image


def analyzeiocr(req):
    if req.FILES:
        img_res = {}
        img_px = []
        for file in req.FILES.values():
            im = PIL.Image.open(file)
            bounds = reader.readtext(im)
            img2 = draw_boxes(im, bounds)
            ress = []
            for i in range(len(bounds)):
                ress.append(bounds[i][1])
            pix = np.array(img2)
            imencoded = cv2.imencode("hello.jpg", pix)[1]
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'", '')
            with open('newfile.txt', 'w') as file:
                file.write(str(img_res))
            img_px.append(ress)
    img_res['px_details'] = img_px
    return JsonResponse(img_res)


def analyzeobjectcount(req):
    class_names = []
    with open("coco2.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    if req.FILES:
        img_res = {}
        img_px = []
        for file in req.FILES.values():
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            colors = np.random.uniform(0, 255, size=(len(class_names), 3))
            net = cv2.dnn.readNet("yolov4-csp.weights", "yolov4-csp.cfg")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            model = cv2.dnn_DetectionModel(net)
            model.setInputParams(size=(512, 512), scale=1 / 255, swapRB=True)
            classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            for (classid, score, box) in zip(classes, scores, boxes):
                color = colors[classid[0]]
                label = "%s : %f" % (class_names[classid[0]], score)
                cv2.rectangle(img, box, color, 2)
                cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            imencoded = cv2.imencode("hello.jpg", img)[1]
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'", '')
            with open('newfile.txt', 'w') as file:
                file.write(str(img_res))
            img_px.append(len(boxes))
    img_res['px_details'] = img_px
    return JsonResponse(img_res)


def analyzefacebeauty(req):
    makeup_a = 'vFG137.png'
    makeup_b = 'vFG112.png'
    # makeup_a = os.path.join('faces', 'makeup', 'vFG137.png')
    # makeup_b = os.path.join('faces', 'makeup', 'vFG112.png')
    img_size = 256
    if req.FILES:
        img_res = {}
        img_px = []
        for file in req.FILES.values():
            img = file
            model = DMT(makeup_a, makeup_b, img, img_size)
            model.load_model()
            model.pairwise(img, makeup_a)
            model.interpolated(img, makeup_a)
            model.hybrid(img, makeup_a, makeup_b)
            model.multimodal(file)
            img2 = imread(os.path.join('output', 'multimodal.jpg'))
            img3 = np.array(img2)
            imencoded = cv2.imencode("hello.jpg", img3)[1]
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'", '')
            with open('newfile.txt', 'w') as file:
                file.write(str(img_res))
            img_px.append("0")
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
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 70, 70])
            upper_red = np.array([20, 200, 150])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
            lower_red = np.array([170, 70, 70])
            upper_red = np.array([180, 200, 150])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
            mask = mask0 + mask1
            img_px.append(np.sum(mask) / 255)
            al = cv2.bitwise_and(img, img, mask=mask)
            dst = cv2.addWeighted(img, 0.1, al, 0.9, 0)
            imencoded = cv2.imencode("hello.jpg", dst)[1]
            img_res[str(file)] = str(base64.b64encode(imencoded.tostring())).strip("b/").replace("'", '')
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


class DMT(object):
    def __init__(self, makeup_a, makeup_b, file, img_size):
        self.makeup_a = makeup_a
        self.makeup_b = makeup_b
        self.img_size = img_size
        self.pb = 'dmt.pb'
        self.style_dim = 8

    def preprocess(self, img):
        return (img / 255. - 0.5) * 2

    def deprocess(self, img):
        return (img + 1) / 2

    def load_image(self, imginp):
        imagee = PIL.Image.open(imginp)
        path = np.array(imagee)
        img = cv2.resize(path, (self.img_size, self.img_size))
        img_ = np.expand_dims(self.preprocess(img), 0)
        return img / 255., img_


    def load_model(self):
        with tf.Graph().as_default():
            output_graph_def = tf.compat.v1.GraphDef()

            with open(self.pb, 'rb') as fr:
                output_graph_def.ParseFromString(fr.read())
                tf.import_graph_def(output_graph_def, name='')

            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
            graph = tf.compat.v1.get_default_graph()
            self.X = graph.get_tensor_by_name('X:0')
            self.Y = graph.get_tensor_by_name('Y:0')
            self.S = graph.get_tensor_by_name('S:0')
            self.X_content = graph.get_tensor_by_name('content_encoder/content_code:0')
            self.X_style = graph.get_tensor_by_name('style_encoder/style_code:0')
            self.Xs = graph.get_tensor_by_name('decoder_1/g:0')
            self.Xf = graph.get_tensor_by_name('decoder_2/g:0')

    def pairwise(self, A, B):
        A_img, A_img_ = self.load_image(A)
        B_img, B_img_ = self.load_image(B)
        Xs_ = self.sess.run(self.Xs, feed_dict={self.X: A_img_, self.Y: B_img_})

        result = np.ones((self.img_size, 3 * self.img_size, 3))
        result[:, :self.img_size] = A_img
        result[:, self.img_size: 2 * self.img_size] = B_img
        result[:, 2 * self.img_size:] = self.deprocess(Xs_)[0]
        imsave(os.path.join('output', 'pairwise.jpg'), result)

    def interpolated(self, A, B, n=3):
        A_img, A_img_ = self.load_image(A)
        B_img, B_img_ = self.load_image(B)
        A_style = self.sess.run(self.X_style, feed_dict={self.X: A_img_})
        B_style = self.sess.run(self.X_style, feed_dict={self.X: B_img_})

        result = np.ones((self.img_size, (n + 3) * self.img_size, 3))
        result[:, :self.img_size] = A_img
        result[:, (n + 2) * self.img_size:] = B_img

        for i in range(n + 1):
            Xf_ = self.sess.run(self.Xf, feed_dict={self.X: A_img_, self.S: (n - i) / n * A_style + i / n * B_style})
            result[:, (i + 1) * self.img_size: (i + 2) * self.img_size] = self.deprocess(Xf_)[0]
        imsave(os.path.join('output', 'interpolated.jpg'), result)

    def hybrid(self, A, B1, B2, n=3):
        A_img, A_img_ = self.load_image(A)
        B1_img, B1_img_ = self.load_image(B1)
        B2_img, B2_img_ = self.load_image(B2)
        B1_style = self.sess.run(self.X_style, feed_dict={self.X: B1_img_})
        B2_style = self.sess.run(self.X_style, feed_dict={self.X: B2_img_})

        result = np.ones((self.img_size, (n + 3) * self.img_size, 3))
        result[:, :self.img_size] = B1_img
        result[:, (n + 2) * self.img_size:] = B2_img

        for i in range(n + 1):
            Xf_ = self.sess.run(self.Xf, feed_dict={self.X: A_img_, self.S: (n - i) / n * B1_style + i / n * B2_style})
            result[:, (i + 1) * self.img_size: (i + 2) * self.img_size] = self.deprocess(Xf_)[0]
        imsave(os.path.join('output', 'hybrid.jpg'), result)

    def multimodal(self, A, n=3):
        A_img, A_img_ = self.load_image(A)
        limits = [
            [0.21629652, -0.43972224],
            [0.15712686, -0.44275892],
            [0.36736163, -0.2079917],
            [0.16977102, -0.49441707],
            [0.2893533, -0.25862852],
            [0.69064325, -0.11329838],
            [0.31735066, -0.48868555],
            [0.50784767, -0.08443227]
        ]
        result = np.ones((n * self.img_size, n * self.img_size, 3))

        for i in range(n):
            for j in range(n):
                S_ = np.ones((1, 1, 1, self.style_dim))
                for k in range(self.style_dim):
                    S_[:, :, :, k] = np.random.uniform(low=limits[k][1], high=limits[k][0])
                Xf_ = self.sess.run(self.Xf, feed_dict={self.X: A_img_, self.S: S_})
                result[i * self.img_size: (i + 1) * self.img_size, j * self.img_size: (j + 1) * self.img_size] = \
                self.deprocess(Xf_)[0]
        imsave(os.path.join('output', 'multimodal.jpg'), result)
