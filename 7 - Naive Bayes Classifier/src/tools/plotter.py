import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd

global count
count = 1


def plot_confusion_matrix(cm, classes, I, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    global count

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Predicted label', fontsize=15)
    plt.xlabel('Real label', fontsize=15)
    plt.tight_layout()

    plt.savefig(f'fig_{I}_{count}.png', dpi=100)
    count = count+1
    plt.show()


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


def get_scores(cm, classes):

    # calculate classes
    scores = dict()

    n_labels = len(classes)

    for ix in range(n_labels):

        tp = cm[ix][ix]
        tn = sum([cm[y][x] for y in range(n_labels) for x in range(n_labels) if y != ix and x != ix])
        fp = sum([cm[ix][id] for id in range(n_labels) if id != ix])
        fn = sum([cm[id][ix] for id in range(n_labels) if id != ix])
        scores[classes[ix]] = [tp, tn, fp, fn]

    return scores


def get_classes(scores, labels, test_size, shuffle):
    classes = [['test_size', 'label', 'precision', 'recall', 'accuracy', 'f1_score', 'shuffle']]

    for sc, la in zip(scores.values(), labels):
        tp, tn, fp, fn = sc

        class_l = []

        precision = np.round((tp/(tp+fp))*100, 2)
        recall = np.round((tp/(tp+fn))*100, 2)
        accuracy = np.round(((tn+tp)/(tn+fp+tp+fn))*100, 2)
        f1_score = np.round((2*(precision*recall)/(precision+recall)), 2)

        precision = precision if not np.isnan(precision) else float(100.0)
        recall = recall if not np.isnan(recall) else float(100.0)
        accuracy = accuracy if not np.isnan(accuracy) else float(100.0)
        f1_score = f1_score if not np.isnan(f1_score) else float(100.0)

        class_l.extend([test_size, la, precision, recall, accuracy, f1_score, shuffle])

        classes.append(class_l)
    return classes
