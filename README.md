# 数据挖掘大作业

**通过数据挖掘实现电信套餐的个性化推荐。**通过多种算法模型构建基于用户消费行为的电信套餐个性化推荐模型，根据用户业务行为画像结果，分析出用户消费习惯及偏好，匹配用户最合适的套餐，提升用户感知，带动用户需求，从而达到用户价值提升的目标。

> 题目链接：<https://github.com/AlexYangLi/DataMining>

## 文件说明

```
DataMining
│   readme.md
│   main.py -- 主程序
|   evaluate.py -- 计算f1-score
│
└───config
│   │   config.py -- 存放各项参数
|   |   logger.py -- 公共日志类
|
└───module -- 各个算法模块
|   |   LoadFile.py
|   |   Bpnn.py
|   |   Knn.py
|   |   DecisionTree.py
|   |   Svm.py
|   |   Bayes.py
|
└───data
|   |   train.csv -- 训练数据集
|   |   test.csv -- 测试数据集
|   |   submit_sample.csv -- 提交示例文件
|
└───logs
|   │   main.2019-10-25.log -- main日志文件，命名方式为：main.日期.log
|   │   ...
|
└───result -- 存放test.csv的预测结果，格式与data/submit_sample.csv相同
|   |   submit_bpnn(50, 50).csv -- 逗号后面有空格
|   |   submit_knn.csv
|   |   submit_decision_tree.csv
|   |   submit_bayes.csv
|   |   submit_svm.csv
|
└───models -- 存放训练后的模型文件（一次训练时间过长，如果有models则不会进行训练）
    |   bpnn.(50, 50).model -- 逗号后有空格
    |   ...
```

## 执行方法

1. 训练与预测

   ```powershell
   python main.py
   ```
   > 进行遗传特征选择算法，通过决策树进行评分，其中最后的submit文件保存为`result/final.csv`

   ```powershell
   python main.py bpnn/knn/bayes/decision_tree/svm
   ```

   > 参数5选一，直接通过以上五种算法进行训练测试，submit文件保存在`result`目录下

2. 评估

   ```powershell
   python evaluate.py --true ./data/test.csv --pred "./result/submit_bpnn(50, 50).csv"
   ```

   > `submit_bpnn(50, 50).csv`中有空格所以需要用引号包裹
   

## 依赖库

* sklearn
* numpy
* logging
* pandas
* joblib
* argparse