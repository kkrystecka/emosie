import numpy as np
import matplotlib.pyplot as plt


def plot_decision(X, y, clf=None, cm=None):
    assert X.ndim == 2

    if clf is not None:
        if cm is None:
            cm = plt.cm.viridis
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        cnt = plt.contourf(xx, yy, Z, 12, cmap=cm)
        for c in cnt.collections:
            c.set_edgecolor("face")
        plt.contour(xx, yy, Z, levels=[0.5])

    for cls in np.unique(y):
        this_class = y == cls
        plt.scatter(X[this_class, 0], X[this_class, 1],
                    edgecolor='k')

    if clf is not None:
        pred = clf.predict(X)
        corr = (pred == y).mean()
        plt.title('correcntess = {}'.format(corr))
