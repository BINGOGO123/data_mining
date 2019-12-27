# -*- coding:utf-8 -*-
# author:臧海彬
# 计算f1-measure

import argparse

import pandas as pd
from sklearn.metrics import f1_score
import ipdb

def score(label,prediction):
    # ipdb.set_trace()
    true_data = pd.DataFrame(label.copy())
    pred_data = pd.DataFrame(prediction.copy())
    current_service2label = dict(zip(sorted(list(set(true_data[0]))),
                                    range(0, len(set(true_data[0])))))
    true_data[0] = true_data[0].map(current_service2label)
    pred_data[0] = pred_data[0].map(current_service2label)

    y_true, y_pred = [], []
    for idx, row in true_data.iterrows():
        y_true.append(row[0])

    for idx, row in pred_data.iterrows():
        y_pred.append(row[0])

    return f1_score(y_true=y_true, y_pred=y_pred, average='macro')
