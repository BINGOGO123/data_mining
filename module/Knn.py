# -*- coding:utf-8 -*-
# author:臧海彬
# 通过knn算法训练以及预测数据

import numpy as np
import sklearn
from sklearn import neighbors
import os
from config.logger import logger
import joblib

class Knn:
    def __init__(self,useExist = False):
        # 首先创建一个分类器
        logger.debug("Knn.__init__")
        self.saveName = "models/knn.model"
        if useExist:
            logger.info("检测knn模型是否已经训练完成")
            if os.path.exists(self.saveName):
                logger.info("knn训练模型已经存在，直接读取训练模型")
                self.clf = joblib.load(self.saveName)
                self.finished = True
            else:
                logger.info("knn训练模型不存在")
                self.clf=neighbors.KNeighborsClassifier()
                self.finished = False
        else:
            logger.info("不检测模型是否存在")
            self.clf=neighbors.KNeighborsClassifier()
            self.finished = False

    # 用训练数据进行拟合
    def train(self,trainData,trainLabel):
        logger.debug("Knn.train")
        self.clf.fit(trainData,trainLabel)
        self.finished = True
    
    # 预测
    def predict(self,testData):
        logger.debug("Knn.predict")
        testPrediction = self.clf.predict(testData)
        return testPrediction

    # 保存训练模型
    def save(self):
        # 如果不存在models文件夹则创建
        if not os.path.exists("models"):
            os.mkdir("models")
        joblib.dump(self.clf,self.saveName)
        logger.info("knn模型已经保存至 {} 中".format(self.saveName))
