import numpy as np
import itertools


def makeTransients(N, A_min = 60, A_max = 100,
                   B_min = -0.5, B_max = 0.0, t1_min = 5, t1_max = 50,
                   trise_min = 3, trise_max = 10, tfall_min = 5, tfall_max = 80):

    A = np.random.uniform(A_min, A_max, size=N)
    B = np.random.uniform(B_min, B_max, size=N)
    t1 = np.random.uniform(t1_min, t1_max, size=N)
    trise = np.random.uniform(trise_min, trise_max, size=N)
    tfall = np.random.uniform(tfall_min, tfall_max, size=N)
    t0 = np.random.uniform(-10, 0, size=N)
    C = [0] * N

    return A, B, C, t0, t1, trise, tfall


def makeTransientLC(theta):
	A, B, C, t0, t1, trise, tfall = theta
	return lambda t: ( (A + B * (t-t0))/(1. + np.exp(-(t-t0)/trise)) + C ) * (t<t1) + \
                     ( (A + B * (t1-t0)) * np.exp(-(t-t1)/tfall) / \
                       (1. + np.exp(-(t-t0)/trise)) + C ) * (t >= t1)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

	From tutorial: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')
    plt.axis('equal')
    print('test2')
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')