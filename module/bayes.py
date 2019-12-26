# -*- coding:utf-8 -*-
# author:臧海彬
# 通过高斯朴素贝叶斯算法以及预测数据

import sklearn
from sklearn.naive_bayes import GaussianNB
from config.logger import logger
import joblib
import os

class Bayes:
    def __init__(self,useExist = False):
        # 首先创建一个分类器
        logger.debug("Bayes.__init__")
        self.saveName = "models/bayes.model"
        if useExist:
            logger.info("检测bayes模型是否已经训练完成")
            if os.path.exists(self.saveName):
                logger.info("bayes训练模型已经存在，直接读取训练模型")
                self.clf = joblib.load(self.saveName)
                self.finished = True
            else:
                logger.info("bayes训练模型不存在")
                self.clf=GaussianNB()
                self.finished = False
        else:
            logger.info("不检测模型是否存在")
            self.clf=GaussianNB()
            self.finished = False

    # 用训练数据进行拟合
    def train(self,trainData,trainLabel):
        logger.debug("Bayes.train")
        self.clf.fit(trainData,trainLabel)
        self.finished = True

    # 预测
    def predict(self,testData):
        logger.debug("Bayes.predict")
        testPrediction = self.clf.predict(testData)
        return testPrediction

    # 保存训练模型（因为一次训练时间太久了）
    def save(self):
        # 如果不存在models文件夹则创建
        if not os.path.exists("models"):
            os.mkdir("models")
        joblib.dump(self.clf,self.saveName)
        logger.info("bayes模型已经保存至 {} 中".format(self.saveName))