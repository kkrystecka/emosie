import os
import numpy as np
import matplotlib.pyplot as plt


def plot_decision(X, y, clf=None, cm=None):
    assert X.ndim == 2

    if clf is not None:
        # choose colormap if not given
        if cm is None:
            cm = plt.cm.viridis

        # create a grid of points to check predictions for
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        # check predictions
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        elif hasattr(clf, 'output_layers'):
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # put the result into a contour plot
        Z = Z.reshape(xx.shape)
        cnt = plt.contourf(xx, yy, Z, 12, cmap=cm)
        for c in cnt.collections:
            c.set_edgecolor("face")
        plt.contour(xx, yy, Z, levels=[0.5])

    # create scatterplot for all classes
    for cls in np.unique(y):
        this_class = y == cls
        plt.scatter(X[this_class, 0], X[this_class, 1],
                    edgecolor='k')

    # add correctness
    if clf is not None:
        pred = clf.predict(X)
        if hasattr(clf, 'output_layers'):
            pred = (pred.ravel() > 0.5).astype('int')
        corr = (pred == y).mean()
        plt.title('correcntess = {}'.format(corr))


def load_images(img_dir, n_images=1000, resize=(50, 50)):
    from keras.preprocessing.image import load_img, img_to_array

    images = os.listdir(img_dir)
    czy_pies = np.array(['dog' in img for img in images])
    n_per_categ = n_images // 2

    n_stars = 0
    imgs, y = list(), list()
    for flt_idx, flt in enumerate([~czy_pies, czy_pies]):
        sel_images = np.array(images)[flt]
        np.random.shuffle(sel_images)
        for idx in range(n_per_categ):
            full_img_path = os.path.join(img_dir, sel_images[idx])
            imgs.append(img_to_array(load_img(full_img_path,
                                              target_size=resize)))
            y.append(flt_idx)

            # progressbar
            if idx % 20 == 0:
                print('*', end='')
                n_stars += 1
            if n_stars == 50:
                n_stars = 0
                print('')

    y = np.array(y)
    imgs = np.stack(imgs, axis=0)
    return imgs, y
