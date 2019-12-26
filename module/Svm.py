# -*- coding:utf-8 -*-
# author:臧海彬
# 通过支持向量机的rbf核训练以及预测数据

from sklearn import svm
import numpy as np
import os
from config.logger import logger
import joblib

class Svm:
    def __init__(self,useExist = False):
        # 首先创建一个分类器
        logger.debug("Svm.__init__")
        self.saveName = "models/svm.model"
        if useExist:
            logger.info("检测Svm模型是否已经训练完成")
            if os.path.exists(self.saveName):
                logger.info("svm训练模型已经存在，直接读取训练模型")
                self.clf = joblib.load(self.saveName)
                self.finished = True
            else:
                logger.info("svm训练模型不存在")
                # svm_clsf=svm.SVC(kernel="rbf",verbose=0,gamma="auto")
                self.svm_clsf=svm.SVC(C=5.0,kernel="rbf",verbose=0,gamma=0.001)
                self.finished = False
        else:
            logger.info("不检测模型是否存在")
            self.svm_clsf=svm.SVC(C=5.0,kernel="rbf",verbose=0,gamma=0.001)
            self.finished = False

    # 用训练数据进行拟合
    def train(self,trainData,trainLabel):
        logger.debug("Svm.train")
        self.clf=self.svm_clsf.fit(trainData,trainLabel)
        self.finished = True
    
    # 预测
    def predict(self,testData):
        logger.debug("Svm.predict")
        testPrediction = self.clf.predict(testData)
        return testPrediction

    # 保存训练模型
    def save(self):
        # 如果不存在models文件夹则创建
        if not os.path.exists("models"):
            os.mkdir("models")
        joblib.dump(self.clf,self.saveName)
        logger.info("svm模型已经保存至 {} 中".format(self.saveName))