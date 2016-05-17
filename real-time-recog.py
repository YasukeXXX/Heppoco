# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers , Variable
from chainer import serializers
from chainer import link
import sys

import cPickle as pickle
import os
import threading
import PIL.Image as Image
import cv2

func = pickle.load(open('googlenet.pkl', 'rb'))
synset_words = np.array([line[10:-1] for line in open(os.path.join('ilsvrc', 'synset_words.txt'), 'r')])

in_size = 224
cropwidth = 256 - in_size
start = cropwidth // 2
stop = start + in_size

# 平均画像の読み込み
meanfile = os.path.join('ilsvrc', 'ilsvrc_2012_mean.npy')
mean_image = np.load(meanfile)
mean_image = mean_image[:, start:stop, start:stop].copy()

def predict(image):
	"""画像を判別"""
	global mean_image, in_size, cropwidth, start, stop
	
	def swap(x):
		x = np.array(x)[:, :, ::-1]
		x = np.swapaxes(x, 0, 2)
		x = np.swapaxes(x, 1, 2)
		return x

	x_data = np.ndarray((1, 3, in_size, in_size), dtype=np.float32)
	
	image = swap(image)
	image = image[:, start:stop, start:stop].copy().astype(np.float32)
	x_data[0] = image-mean_image

	x = chainer.Variable(x_data, volatile=True)
	
	y, = func(inputs={'data': x}, outputs=['loss3/classifier'], train=False)
	synset_i = y.data.argmax(axis=1)
	return synset_words[synset_i]

def cap():
    """リアルタイムで画像をキャプチャする"""
    global capture,msg
    fontface=cv2.FONT_ITALIC
    fontscale=1.0
    bold = 4
    color=(255,0,255)
    
    while True:
        ret, image = capture.read()

        if ret == False:
            cv2.destroyAllWindows()
            break

        location=(0,image.shape[0]/2)
        cv2.putText(image,msg,location,fontface,fontscale,color,bold)  
        cv2.imshow("Image Recognition", image)

        if cv2.waitKey(33) >= 0:
    #         cv2.imwrite("image.png", image)
            cv2.destroyAllWindows()
            break
    print 'out'
    cv2.destroyAllWindows()

def main():
    """取り込まれた画像を判別してラベルをセット"""
    global msg, capture
    while True:
        ret, image = capture.read()

        if ret == False:
            break
        img = Image.fromarray(image[::-1, :, ::-1].copy()).resize((256, 256), Image.ANTIALIAS)

        # ラベルをcapとの共通変数にセット
        msg = predict(img)[0]

        # キー入力で終了
        if cv2.waitKey(33) >= 0:
            break
        if capThread.isAlive() == False:
            break

capture = cv2.VideoCapture(0)
msg = ''
cv2.namedWindow("Image Recognition", cv2.WINDOW_NORMAL)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,324)  #2592
capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,228) #1944
capThread = threading.Thread(target=cap, name='cap')
capThread.setDaemon(True)
capThread.start()
main()
