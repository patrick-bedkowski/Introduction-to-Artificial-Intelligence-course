import copy
import sys

import pandas as pd
import numpy as np
import tabulate

from sklearn.model_selection import KFold, train_test_split
from plotter import plot_confusion_matrix

from tree import ID3


def make_folds(df, label, _shuffle):

    X = df.loc[:, df.columns != label]
    y = df.loc[:, df.columns == label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=_shuffle, test_size = 0.8, random_state = 0)  #

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    return df_train, df_test


def better_confusion_matrix(y_true, y_pred, labels):

    n_labels = len(labels)
    output = np.zeros(shape=(n_labels, n_labels), dtype=int)

    new_df = pd.concat([y_true, y_pred], axis=1)
    results = dict(new_df.value_counts())

    label_idx = {labels[idx]: idx for idx in range(0, len(labels))}

    for x, quantity in results.items():
        real_label = x[0]
        pred_label = x[1]

        real_idx = label_idx[real_label]
        pred_idx = label_idx[pred_label]

        output[pred_idx][real_idx] = int(quantity)

    return output


def get_scores(cm, n_labels):

    # calculate classes
    scores = []

    for ix in range(n_labels):

        tp = cm[ix][ix]
        tn = sum([cm[y][x] for y in range(n_labels) for x in range(n_labels) if y != ix and x != ix])
        fp = sum([cm[ix][id] for id in range(n_labels) if id != ix])
        fn = sum([cm[id][ix] for id in range(n_labels) if id != ix])
        scores.append([tp, tn, fp, fn])

    return scores


def get_classes(scores, labels, tt_mode=False):
    classes = [['Label', 'Precision', 'Recall', 'Accuracy', 'F1 score']]

    for sc, la in zip(scores.values(), labels):
        if tt_mode:
            tp, tn, fp, fn, train_r, test_r = sc
        else:
            tp, tn, fp, fn = sc

        class_l = []

        precision = round((tp/(tp+fp))*100, 2)
        recall = round((tp/(tp+fn))*100, 2)
        accuracy = round(((tn+tp)/(tn+fp+tp+fn))*100, 2)
        f1_score = round((2*(precision*recall)/(precision+recall)), 2)

        precision = precision if not np.isnan(precision) else float(0.0)
        recall = recall if not np.isnan(recall) else float(0.0)
        accuracy = accuracy if not np.isnan(accuracy) else float(0.0)
        f1_score = f1_score if not np.isnan(f1_score) else float(0.0)

        if tt_mode:
            class_l.extend([la, precision, recall, accuracy, f1_score, train_r, test_r])
        else:
            class_l.extend([la, precision, recall, accuracy, f1_score])

        classes.append(class_l)
    return classes


def average_scores(scores_classes, k_folds):
    agg_scores = [['Label', 'Precision', 'Recall', 'Accuracy', 'F1 score']]
    dicts = {label: 0 for label in agg_scores[0][1:]}
    for score in scores_classes:
        for label, label_name in zip(score[1:], agg_scores[0][1:]):
            dicts[label_name] += 1
            if label[0] not in [row[0] for row in agg_scores]:
                agg_scores.append(label)
            else:
                for label_agg_idx in range(len([x[0] for x in agg_scores])):
                    if agg_scores[label_agg_idx][0] == label[0]:
                        for i in range(1, 5):
                            agg_scores[label_agg_idx][i] += label[i]
    # average
    for i in range(1, len(agg_scores)):
        for j, label_name in zip(range(1, 5), agg_scores[0][1:]):
            agg_scores[i][j] = round(agg_scores[i][j]/dicts[label_name], 2)

    return agg_scores


def scores_to_dataframe_ttsplit(avg_scores):
    columns = avg_scores[0][0] + ['train_r', 'test_r']
    data = [y for x in avg_scores for y in x[1:]]
    return pd.DataFrame(data=data, columns=columns)


def scores_to_dataframe_kfolds(avg_scores, k_folds):
    columns = avg_scores[0][1:] + ['Folds']
    index = [x[0] for x in avg_scores[1:]]
    data = [[y for y in x[1:]] + [k_folds] for x in avg_scores[1:]]

    return pd.DataFrame(data=data, columns=columns, index=index)


def run_tt_split(df, tt_r, single=True):
    all_scores_df = None
    scores_classes = []

    if single:
        tt_r = [tt_r[0]]

    print('Running train split ratio test.')

    for test_r, train_r in tt_r:
        model = ID3()

        train_X, test_y = train_test_split(df, test_size=test_r, train_size=train_r, shuffle=False)

        # # fit the model | build the model
        tree = model.fit(train_X, LABEL)

        # predict the testing classes
        df_hat = model.predict(test_y)

        #
        # Process data for scoring
        #
        unique_labels = df_hat['decision'].unique()
        unique_predicted = df_hat['decision_hat'].unique()  # .value_counts().index.to_list()
        unique_all = df['decision'].unique()
        diff = [x for x in unique_labels if x not in unique_predicted]
        real = list(unique_predicted) + diff
        y_test, y_pred = df_hat['decision'], df_hat['decision_hat']

        #
        # Confusion matrix
        #
        cm = better_confusion_matrix(y_test, y_pred, real)
        plot_confusion_matrix(cm, classes=real, I=None)

        #
        # SCORE THE MODEL
        #
        scores = get_scores(cm, len(real))

        # add information about test_r, train_r
        scores = [score + [train_r, test_r] for score in scores]

        classes = get_classes(scores, real, True)

        scores_classes.append(classes)

        print(tabulate.tabulate(classes, tablefmt='fancy_grid'))
    #
    # Aggregate scores
    #
    #print(tabulate.tabulate(scores_classes, tablefmt='fancy_grid'))

    scores_df = scores_to_dataframe_ttsplit(scores_classes)

    if all_scores_df is None:
        all_scores_df = copy.copy(scores_df)
    else:
        all_scores_df = pd.concat([all_scores_df, scores_df], axis=0)

    all_scores_df.to_excel('scores_t_t_split.xlsx', index=True)


def run_k_fold(df, k_folds):
    all_scores_df = None
    scores_classes = []

    print('Running k_fold tests.')

    for t in range(0, 5):
        print(f'Test number: {t}')
        for k_fold in k_folds:
            model = ID3()
            kf = KFold(n_splits=k_fold, shuffle=False)

            for train_index, test_index in kf.split(df):

                train = df.iloc[train_index]
                test = df.iloc[test_index]

                # # fit the model | build the model
                tree = model.fit(train, LABEL)

                # predict the testing classes
                df_hat = model.predict(test)

                #
                # Process data for scoring
                #
                unique_labels = df_hat['decision'].unique()
                unique_predicted = df_hat['decision_hat'].unique()  # .value_counts().index.to_list()
                unique_all = df['decision'].unique()
                diff = [x for x in unique_labels if x not in unique_predicted]
                real = list(unique_predicted) + diff
                y_test, y_pred = df_hat['decision'], df_hat['decision_hat']

                #
                # Confusion matrix
                #
                cm = better_confusion_matrix(y_test, y_pred, real)
                plot_confusion_matrix(cm, I=k_fold, classes=real)

                #
                # SCORE THE MODEL
                #
                scores = get_scores(cm, len(real))
                classes = get_classes(scores, real)

                scores_classes.append(classes)

                # print(tabulate.tabulate(classes, tablefmt='fancy_grid'))

            #
            # Aggregate scores
            #
            avg_scores = average_scores(scores_classes, k_fold)

            print(tabulate.tabulate(avg_scores, tablefmt='fancy_grid'))

            scores_df = scores_to_dataframe_kfolds(avg_scores, k_fold)

            if all_scores_df is None:
                all_scores_df = copy.copy(scores_df)
            else:
                all_scores_df = pd.concat([all_scores_df, scores_df], axis=0)

    all_scores_df.to_excel('scores_kfold.xlsx', index=True)


if __name__ == '__main__':

    TESTING_K_FOLDS = [2, 3, 5, 7, 10, 20]

    TRAIN_TEST_RATIOS = [(round(x, 2), 1.0-(round(x, 2))) for x in np.arange(0.1, 0.6, 0.05)]

    if len(sys.argv) == 1:
        print('Not enough arguments have been passed into program!')
        sys.exit()
    else:
        file_name = sys.argv[1]

        df = pd.read_csv(file_name)

        # assumption
        LABEL = list(df.columns)[-1]  # get last column

        run_tt_split(df, TRAIN_TEST_RATIOS)
