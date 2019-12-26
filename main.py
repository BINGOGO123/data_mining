# -*- coding:utf-8 -*-
# author:臧海彬
# 训练以及测试输出结果主函数
import sys
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import logging
import datetime
from module.LoadFile import LoadFile
from config.config import trainFile
from config.config import testFile
from config.logger import logger
from module.Bpnn import Bpnn
from module.Knn import Knn
from module.Svm import Svm
from module.DecisionTree import DecisionTree
from module.Bayes import Bayes

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

def bayesClassifier(trainData,trainLabel,testData,testDataSet,load):
    logger.info("bayes算法预测开始")
    filename = "result/submit_bayes.csv"
    bayes = Bayes(True)
    if not bayes.finished:
        bayes.train(trainData,trainLabel)
        bayes.save()
        testPrediction = bayes.predict(testData)
        load.write(filename,testDataSet[1],testPrediction)
        logger.info("bayes算法训练结束，predict结果保存在 {} 中".format(filename))
    else:
        logger.info("bayes算法已经预测过，predict结果保存在 {} 中".format(filename))

def decision_treeClassifier(trainData,trainLabel,testData,testDataSet,load):
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
    if len(sys.argv) > 1:
        para = sys.argv[1]
    else:
        para = "bpnn"
    try:
        fun = eval(para + "Classifier")
    except:
        logger.exception("没有 {} 分类器".format(para))
        sys.exit()

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

    # 目前bpnn有点特殊
    if fun is bpnnClassifier:
        size = (100,100)
        fun(size,trainData,trainLabel,testData,testDataSet,load)
    else:
        fun(trainData,trainLabel,testData,testDataSet,load)
