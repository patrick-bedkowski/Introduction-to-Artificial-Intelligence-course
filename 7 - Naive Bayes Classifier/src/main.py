from src.tools.file_management import load_data
from src.tools.plotter import (plot_confusion_matrix,
                               better_confusion_matrix,
                               get_scores,
                               get_classes)
from src.algorithms.NaiveBayesianClassifier import NaiveBayesianClassifier

import pandas as pd

if __name__ == '__main__':
    dataset = load_data()  # import data

    train_size = 0.8
    test_size = 0.2
    shuffle = True

    CLASSES = list(dataset.target.unique())

    # create dataframe for score aggregation
    scores_df = pd.DataFrame(columns=['train_size', 'label',
                                      'precision', 'recall',
                                      'accuracy', 'f1_score',
                                      'shuffle'])

    nbc = NaiveBayesianClassifier(test_size=test_size)
    actual, predictions = nbc.fit_predict(dataset, shuffle=shuffle)

    # create matrix
    cm = better_confusion_matrix(actual, predictions, CLASSES)
    scores = get_scores(cm, CLASSES)

    classes_with_scores = get_classes(scores, CLASSES, train_size, str(shuffle))

    plot_confusion_matrix(cm, classes=CLASSES, I=None)
    columns = classes_with_scores[0]
    for row in classes_with_scores[1:]:
        to_append = dict(zip(columns, row))
        scores_df = scores_df.append(to_append, ignore_index=True)

    print(scores_df)
