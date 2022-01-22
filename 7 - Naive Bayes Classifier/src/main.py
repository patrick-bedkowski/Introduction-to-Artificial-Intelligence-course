from src.tools.file_management import load_data
from src.tools.plotter import (plot_confusion_matrix,
                               better_confusion_matrix,
                               get_scores,
                               get_classes)
from src.algorithms.NaiveBayesianClassifier import NaiveBayesianClassifier

import numpy as np
import pandas as pd

if __name__ == '__main__':
    dataset = load_data()

    test_sizes = list(np.arange(0.1, 0.8, 0.1))
    shuffles = [True, False]

    CLASSES = list(dataset.target.unique())

    # create dataframe for score aggregation
    scores_df = pd.DataFrame(columns=['test_size', 'label',
                                      'precision', 'recall',
                                      'accuracy', 'f1_score',
                                      'shuffle'])


    for shuffle in shuffles:
        for test_size in test_sizes:
            nbc = NaiveBayesianClassifier(test_size=test_size)
            actual, predictions = nbc.fit_predict(dataset, shuffle=shuffle)

            # create matrix
            cm = better_confusion_matrix(actual, predictions, CLASSES)
            scores = get_scores(cm, CLASSES)

            classes_with_scores = get_classes(scores, CLASSES, test_size, str(shuffle))

            # plot_confusion_matrix(cm, classes=CLASSES, I=None)
            columns = classes_with_scores[0]
            for row in classes_with_scores[1:]:
                to_append = dict(zip(columns, row))
                scores_df = scores_df.append(to_append, ignore_index=True)

    # sort data
    scores_df.sort_values(["label", "test_size"],
                          axis=0, ascending=True,
                          inplace=True)

    scores_df.to_excel('scores_1.xlsx', index=True)
