import matplotlib.pyplot as plt
import numpy as np
import itertools

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
