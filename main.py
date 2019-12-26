# -*- coding:utf-8 -*-
# author:臧海彬
# 训练以及测试输出结果主函数
import sys
import numpy as np
import os
from module.LoadFile import LoadFile
from config.config import trainFile
from config.config import testFile
from sklearn.preprocessing import StandardScaler
import logging
from config.logger import logger
from module.Bpnn import Bpnn
import datetime
from module.Knn import Knn
from module.Svm import Svm
from module.DecisionTree import DecisionTree

os.chdir(sys.path[0])

# 初始化日志对象
def initLogger():
    # 如果不存在logs文件夹则创建
    if not os.path.exists("logs"):
        os.mkdir("logs")
    handler1 = logging.FileHandler("logs/main." + str(datetime.date.today()) + ".log","a",encoding="utf8")
    handler2 = logging.StreamHandler()
    formatter1 = logging.Formatter(fmt="%(asctime)s [%(levelname)s] [%(lineno)d] >> %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    formatter2 = logging.Formatter(fmt = "[%(levelname)s] >> %(message)s")
    handler1.setFormatter(formatter1)
    handler2.setFormatter(formatter2)
    handler1.setLevel(logging.DEBUG)
    handler2.setLevel(logging.DEBUG)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

# 数据预处理
def scaler(trainData,testData):
    logger.debug("scaler")
    # 但是这里归一化操作不用测试数据训练，因为题目要求这里还不知道测试数据的
    scaler = StandardScaler() # 标准化转换
    scaler.fit(trainData)  # 只用训练集训练标准化对象
    return scaler.transform(trainData),scaler.transform(testData)

def bpnnClassifier(size,trainData,trainLabel,testData,testDataSet,load):
    logger.info("神经网络算法预测开始，规模：{}".format(size))
    filename = "result/submit_bpnn" + str(size) + ".csv"
    bpnn = Bpnn(size,True)
    if not bpnn.finished:
        bpnn.train(trainData,trainLabel)
        bpnn.save()
        bpnn.print()
        testPrediction = bpnn.predict(testData)
        load.write(filename,testDataSet[1],testPrediction)
        logger.info("神经网络训练测试结束，predict结果保存在 {} 中".format(filename))
    else:
        logger.info("{} 规模神经网络已经预测过，predict结果保存在 {} 中".format(size,filename))
        # testPrediction = bpnn.predict(testData)
        # load.write(filename,testDataSet[1],testPrediction)
        # logger.info("神经网络训练测试结束，predict结果保存在 {} 中".format(filename))

def knnClassifier(trainData,trainLabel,testData,testDataSet,load):
    logger.info("Knn算法预测开始")
    filename = "result/submit_knn.csv"
    knn = Knn(True)
    if not knn.finished:
        knn.train(trainData,trainLabel)
        knn.save()
        testPrediction = knn.predict(testData)
        load.write(filename,testDataSet[1],testPrediction)
        logger.info("knn算法训练结束，predict结果保存在 {} 中".format(filename))
    else:
        logger.info("knn算法已经预测过，predict结果保存在 {} 中".format(filename))

def svmClassifier(trainData,trainLabel,testData,testDataSet,load):
    logger.info("svm算法预测开始")
    filename = "result/submit_svm.csv"
    svm = Svm(True)
    if not svm.finished:
        svm.train(trainData,trainLabel)
        svm.save()
        testPrediction = svm.predict(testData)
        load.write(filename,testDataSet[1],testPrediction)
        logger.info("svm算法训练结束，predict结果保存在 {} 中".format(filename))
    else:
        logger.info("svm算法已经预测过，predict结果保存在 {} 中".format(filename))

def decisionTreeClassifier(trainData,trainLabel,testData,testDataSet,load):
    logger.info("decision_tree算法预测开始")
    filename = "result/submit_decision_tree.csv"
    decisionTree = DecisionTree(True)
    if not decisionTree.finished:
        decisionTree.train(trainData,trainLabel)
        decisionTree.save()
        testPrediction = decisionTree.predict(testData)
        load.write(filename,testDataSet[1],testPrediction)
        logger.info("decision_tree算法训练结束，predict结果保存在 {} 中".format(filename))
    else:
        logger.info("decision_tree算法已经预测过，predict结果保存在 {} 中".format(filename))

if __name__ == "__main__":
    initLogger()
    # 读取数据
    load = LoadFile()
    trainDataSet = load.read(trainFile)
    trainLabel = trainDataSet[0][:,-1].astype(int)
    trainData = trainDataSet[0][:,:-1]
    testDataSet = load.read(testFile)
    testLabel = testDataSet[0][:,-1].astype(int)
    testData = testDataSet[0][:,:-1]
    # 神经网络对数据尺度敏感，所以最好在训练前标准化，或者归一化，或者缩放到[-1,1]
    trainData,testData = scaler(trainData,testData)

    # 神经网络训练
    # size = (50,50)
    # bpnnClassifier(size,trainData,trainLabel,testData,testDataSet,load)

    # k近邻算法
    # knnClassifier(trainData,trainLabel,testData,testDataSet,load)

    # 支持向量机算法
    svmClassifier(trainData,trainLabel,testData,testDataSet,load)

    # 决策树算法
    # decisionTreeClassifier(trainData,trainLabel,testData,testDataSet,load)
