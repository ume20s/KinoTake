# -*- coding: utf-8 -*-
#from __future__ import print_function
import argparse
import os
import sys
import cv2
import numpy as np
from PIL import Image


import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

# 畳み込みニューラルネットワークの定義
class CNN(chainer.Chain):
    def __init__(self, n_units, n_out):
        w = I.Normal(scale=0.05) # モデルパラメータの初期化
        super(CNN, self).__init__(                        
            conv1=L.Convolution2D(3, 16, 5, 1, 0), # 1層目の畳み込み層（フィルタ数は16）
            conv2=L.Convolution2D(16, 32, 5, 1, 0), # 2層目の畳み込み層（フィルタ数は32）
            conv3=L.Convolution2D(32, 64, 5, 1, 0), # 3層目の畳み込み層（フィルタ数は64）
            l4=L.Linear(None, n_out, initialW=w), #クラス分類用
        )
    
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2)
        # 9x9,64ch
        return self.l4(h3)


def crop(img, size):
    w, h = img.size
    assert w >= 100 or h >= 100, "画像サイズが小さすぎます"    
    p = (w - h) / 2 if w > h else (h - w) / 2
    box = (p, 0, p+h, h)
    return img.crop(box).resize((size, size))

def main():
    # オプション処理
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--model', '-m', default='model',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    model = L.Classifier(CNN(args.unit, 2)) 
    chainer.serializers.load_npz(args.model, model)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        img = Image.fromarray(frame)
        img = np.asarray(crop(img, 100))
        img = img.transpose(2, 0, 1)
        x = chainer.Variable(np.array([img]).astype(np.float32))
        y = model.predictor(x)
        c = F.softmax(y).data.argmax()
        s = F.softmax(y) 
        hantei = '「きのこの山」' if c == 1 else '「たけのこの里」'
        print('判定結果は{}です。 {}'.format(hantei, s.data)) 

        #本来なら以下のコードでESC入力でループを抜けるが，抜けないのでctrl-cでプログラムを終了してください
        #k = cv2.waitKey(1000) 
        #print k
        #if k == 27:
        #    break
                        
    cap.release()
    
    
if __name__ == '__main__':
    main()
