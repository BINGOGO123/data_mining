# -*- coding:utf-8 -*-
# author:臧海彬
# 从指定的csv文件中读取数据,以及将结果写入到csv文件中
from config.logger import logger
import numpy as np
from csv import reader

class LoadFile:
    def read(self,filename):
        logger.debug("LoadFile.read({})".format(filename))
        file = open(filename, "rt")
        lines = reader(file)
        list_lines=list(lines)
        file.close()
        # 去掉第一行的标签
        list_lines = list_lines[1:]
        # 其中第21、22维数据（性别、年龄可能是未知，本来标作\N,这里置为0表示未知，方便存储）
        for i in range(len(list_lines)):
            for j in range(len(list_lines[0])):
                if list_lines[i][j] == "\\N":
                    list_lines[i][j] = "0"
        data1 = []
        data2 = []
        for x in list_lines:
            data1.append(x[:-1])
            data2.append(x[-1])
        return np.array(data1,np.float64),np.array(data2)
    
    def write(self,filename,userId,prediction):
        logger.debug("LoadFile.write({})".format(filename))
        file = open(filename,"w")
        file.write("user_id,current_service\n")
        for x,y in zip(userId,prediction.astype(int)):
            file.write("{},{}\n".format(x,y))
