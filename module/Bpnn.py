# -*- coding:utf-8 -*-
# author:臧海彬
# 通过bp神经网络训练以及预测数据

import sklearn
from sklearn.neural_network import MLPClassifier
from config.logger import logger
import joblib
import os

class Bpnn:
    def __init__(self,size,useExist = False):
        # 首先创建一个分类器
        logger.debug("Bpnn.__init__({})".format(size))
        self.size = size
        self.saveName = "models/bpnn." + str(self.size) + ".model"
        if useExist:
            logger.info("检测 {} 模型是否已经训练完成".format(self.size))
            if os.path.exists(self.saveName):
                logger.info("规模:{} 训练模型已经存在，直接读取训练模型".format(self.size))
                self.clf = joblib.load(self.saveName)
                self.finished = True
            else:
                logger.info("规模:{} 训练模型不存在".format(self.size))
                self.clf = MLPClassifier(hidden_layer_sizes=size)
                self.finished = False
        else:
            logger.info("不检测模型是否存在")
            self.clf = MLPClassifier(hidden_layer_sizes=size)
            self.finished = False

    # 用训练数据进行拟合
    def train(self,trainData,trainLabel):
        logger.debug("Bpnn.train")
        self.clf.fit(trainData,trainLabel)
        self.finished = True
    
    # 打印一下结果
    def print(self):
        logger.debug("Bpnn.print")
        cengindex=0
        for wi in self.clf.coefs_:
            cengindex += 1  # 表示底第几层神经网络。
            logger.info("第{}层网络层: 权重矩阵维度={} 系数矩阵={}:".format(cengindex,wi.shape,wi))

    # 预测
    def predict(self,testData):
        logger.debug("Bpnn.predict")
        testPrediction = self.clf.predict(testData)
        return testPrediction

    # 保存训练模型（因为一次训练时间太久了）
    def save(self):
        # 如果不存在models文件夹则创建
        if not os.path.exists("models"):
            os.mkdir("models")
        joblib.dump(self.clf,self.saveName)
        logger.info("神经网络模型已经保存至 {} 中".format(self.saveName))