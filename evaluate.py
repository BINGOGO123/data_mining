# -*- coding: utf-8 -*-

import argparse

import pandas as pd
from sklearn.metrics import f1_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--true', type=str, required=True, help='path to ground truth file')
    parser.add_argument('--pred', type=str, required=True, help='path to predictions file')

    args = parser.parse_args()

    true_data = pd.read_csv(args.true, header=0, index_col=None)
    pred_data = pd.read_csv(args.pred, header=0, index_col=None)
    current_service2label = dict(zip(sorted(list(set(true_data['current_service']))),
                                     range(0, len(set(true_data['current_service'])))))

    true_data['current_service'] = true_data['current_service'].map(current_service2label)
    pred_data['current_service'] = pred_data['current_service'].map(current_service2label)

    user_to_true_label = dict()
    for idx, row in true_data.iterrows():
        user_to_true_label[row['user_id']] = row['current_service']

    user_to_pred_label = dict()
    for idx, row in pred_data.iterrows():
        user_to_pred_label[row['user_id']] = row['current_service']

    y_true, y_pred = [], []
    for user_id, true_label in user_to_true_label.items():
        if user_id not in user_to_pred_label:
            raise Exception(f'user id not in prediction file: {user_id}')
        y_true.append(true_label)
        y_pred.append(user_to_pred_label[user_id])

    print('F1 score:', f1_score(y_true=y_true, y_pred=y_pred, average='macro'))





