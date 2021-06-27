# -*- coding: utf-8 -*-
#from __future__ import print_function
import argparse
import os
import sys

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

def main():
    # オプション処理
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    train = []
    label = 0
    print('loading dataset')
    for c in os.listdir('data/train'):
        print('class: {}, class id: {}'.format(c, label))
        d = os.path.join('data/train', c)        
        imgs = os.listdir(d)
        for i in [f for f in imgs if ('jpg' in f)]:
            train.append([os.path.join(d, i), label])            
        label += 1    
    train = chainer.datasets.LabeledImageDataset(train, '.')    
    train, test = chainer.datasets.split_dataset_random(train, 1000)        

    model = L.Classifier(CNN(args.unit, 2)) 
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu() 

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # イタレータの定義
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    # 評価データ用のイタレータ
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
        
    # 計算グラフ
    trainer.extend(extensions.dump_graph('main/loss'))
    # スナップショット
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    # ログを出力
    trainer.extend(extensions.LogReport())    
    # エポック経過ごとの損失の値をグラフ表示する
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',
                              file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                              'epoch', file_name='accuracy.png'))
    # 途中結果
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    # プログレスバー
    trainer.extend(extensions.ProgressBar())

    # 途中まで学習していたトレーナーの状態を読み込む
    if args.resume:        
        chainer.serializers.load_npz(args.resume, trainer)

    # 学習開始
    trainer.run()

    # モデルをCPU対応へ
    model.to_cpu()
    # 保存
    modelname = args.out + "/CNN.model"
    print('save the trained model: {}'.format(modelname))
    chainer.serializers.save_npz(modelname, model)

if __name__ == '__main__':
    main()
