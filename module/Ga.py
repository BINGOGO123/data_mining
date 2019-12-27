# -*- coding:utf-8 -*-
# author:臧海彬
# 通过遗传算法进行特征选择

import random
from config.logger import logger

class Life:
    def __init__(self,array):
        self.life = array
    
    def cross(self,life,bianyilv):
        locate = int((len(self.life) - 1) * random.random())
        life = Life(self.life[0:locate + 1] + life.life[locate + 1:])
        if random.random() < bianyilv:
            life.bianyi()
        return life

    def bianyi(self):
        locate = int(len(self.life) * random.random())
        self.life[locate] = 1 - self.life[locate]

class Ga:
    def __init__(self,maxCount,geneLength,scoreFun,bianyilv):
        self.maxCount = maxCount
        self.geneLength = geneLength
        self.scoreFun = scoreFun
        self.bianyilv = bianyilv
        self.currentCount = 0
        self.best = -1
        self.bestLife = None
        logger.info("Ga.__init__(maxCount = {},geneLength = {},scoreFun = {},bianyilv = {})".format(maxCount,geneLength,scoreFun,bianyilv))

        self.lifes = []
        logger.info("--初始种群--")
        for i in range(4):
            life = Life([int(2 * random.random()) for x in range(self.geneLength)])
            logger.info(life.life)
            self.lifes.append(life)

    def go(self):
        scoreList = []
        left = []
        bestLife = -1
        bestScore = -1

        for i in self.lifes:
            thisScore = self.scoreFun(self.numToBool(i.life))
            logger.info("{} : {}".format(i.life,thisScore))
            scoreList.append(thisScore)
            if thisScore > bestScore:
                bestScore = thisScore
                bestLife = i
        self.best = bestScore
        self.bestLife = bestLife
        left.append(bestLife)

        sumScore = 0
        for x in scoreList:
            sumScore += x
        parts = []
        for x in scoreList:
            parts.append(x/sumScore)
        for i in range(len(parts)):
            if i != 0:
                parts[i] += parts[i - 1]
        
        end = False
        while True:
            randomNumber = random.random()
            for j in range(len(parts)):
                if randomNumber <= parts[j]:
                    if self.lifes[j] is left[0]:
                        break
                    else:
                        end = True
                        left.append(self.lifes[j])
                        break
            else:
                if self.lifes[-1] is not left[0]:
                    end = True
                    left.append(self.lifes[-1])

            if end:
                break
        
        for i in range(2):
            left.append(left[0].cross(left[1],self.bianyilv))
        self.lifes = left
        self.currentCount += 1
        logger.info("当前count: {}, 总count：{}, 最优个体：{}, 得分：{}".format(self.currentCount,self.maxCount,self.bestLife.life,self.best))
        logger.info("--当前种群--")
        for life in self.lifes:
            logger.info(life.life)
    
    def start(self):
        logger.info("遗传开始")
        while self.currentCount < self.maxCount:
            self.go()
        logger.info("遗传结束")
        return self.numToBool(self.bestLife.life),self.best

    def numToBool(self,array):
        return [True if x == 1 else False for x in array]
