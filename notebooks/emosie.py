import os
import sys
import tempfile
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


def apply_modifications(model, custom_objects=None):
    """
    Poprawiona wersja apply_modifications biblioteki keras_vis.
    (na githubie jest poprawna wersja ale na pip nie)
    """
    from keras.models import load_model
    fname = next(tempfile._get_candidate_names()) + '.h5'
    model_path = os.path.join(tempfile.gettempdir(), fname)
    model.save(model_path)
    new_model = load_model(model_path, custom_objects=custom_objects)
    os.remove(model_path)
    return new_model


def show_rgb_layers(image, style='light', subplots_args=dict()):
    '''
    Show RGB layers of the image on separate axes.
    '''
    im_shape = image.shape
    assert im_shape[-1] == 3
    assert image.ndim == 3

    if style == 'light':
        cmaps = ['Reds', 'Greens', 'Blues']

    fig, ax = plt.subplots(ncols=3, **subplots_args)
    for layer in range(3):
        if style == 'light':
            ax[layer].imshow(image[..., layer], cmap=cmaps[layer])
        else:
            temp_img = np.zeros(im_shape[:2] + (3,))
            temp_img[..., layer] = image[..., layer]
            ax[layer].imshow(temp_img)
        ax[layer].axis('off')

    return fig


def extract_features(X, model, batch_size=20):
    n_stars = 0
    sample_count = X.shape[0]
    model_shape = (shp.value for shp in model.layers[-1].output.shape[:])
    output_shape = (sample_count,) + tuple(shp for shp in model_shape
                                           if shp is not None)
    features = np.zeros(shape=output_shape)

    n_full_bathes = sample_count // batch_size
    for batch_idx in range(n_full_bathes):
        slc = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)
        features_batch = model.predict(X[slc])
        features[slc] = features_batch

        # progressbar
        print('*', end='')
        n_stars += 1
        if n_stars == 50:
            n_stars = 0
            print('')

    left_out = sample_count - n_full_bathes * batch_size
    if left_out > 0:
        slc = slice(n_full_bathes * batch_size, None)
        features_batch = model.predict(X[slc])
        features[slc] = features_batch

    features = features.reshape((sample_count, -1))
    return features
