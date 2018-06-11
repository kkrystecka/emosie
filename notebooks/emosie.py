import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt


# - [ ] dodać funkcję do wyświetlania błędów i poprawnych
#       odpowiedzi sieci

def plot_decision(X, y, clf=None, cm=None):
    '''
    Plot decision function of a given classifier.

    Parameters
    ----------
    X : 2d numpy array
        Array in classical sklearn format (observations, features).
    y : 1d numpy array
        Correct class membership.
    clf : sklearn classifier or Keras model
        Classifier used in predicting class membership.
    cm : colormap
        Colormap to use for class probabilities.
    '''
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
    '''
    Load images of cats and dogs and organize into sklearn-like format.
    '''
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
    '''
    Use a trained model to extract features from training examples.
    '''
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


def show_image_predictions(X, y, model=None, predictions=None):
    if model is not None and not (predictions is not None):
        predictions = model.predict(X)
    if_correct = np.round(predictions).ravel() == y
    incorrect_predictions = np.where(if_correct == 0)[0]

    # FIXME: change the code below:
    # znajdujemy poprawne przewidywania oraz obliczamy pewność
    confidence = np.abs(predictions.ravel() - 0.5) * 2
    correct_predictions = np.where(if_correct)[0]
    confidence_for_correct_predictions = confidence[correct_predictions]

    # znajdujemy poprawne przedidywania z wysoką pewnością
    high_confidence = np.where(confidence_for_correct_predictions > 0.75)[0]
    correct_high_confidence = correct_predictions[high_confidence]

    # wyświetlamy
    fig, ax = plt.subplots(ncols=6, nrows=3, figsize=(14, 8))
    ax = ax.ravel()

    for idx in range(3 * 6):
        img_idx = correct_high_confidence[idx]
        ax[idx].imshow(X_test[img_idx])
        ax[idx].set_title('{:.2f}%'.format(predictions[img_idx, 0] * 100))
        ax[idx].axis('off')


def test_ipyvolume():
    import ipyvolume as ipv

    s = 1/2**0.5
    # 4 vertices for the tetrahedron
    x = np.array([1.,  -1, 0,  0])
    y = np.array([0,   0, 1., -1])
    z = np.array([-s, -s, s, s])
    # and 4 surfaces (triangles), where the number refer to the vertex index
    triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 3, 2)]

    ipv.figure()
    # draw the tetrahedron mesh
    ipv.plot_trisurf(x, y, z, triangles=triangles, color='orange')
    # mark the vertices
    ipv.scatter(x, y, z, marker='sphere', color='blue')
    # set limits and show
    ipv.xyzlim(-2, 2)
    ipv.show()


def read_brainnetviewer_surface(file):
    '''
    Read surface file stored in BrainNetViewer format.

    Parameters
    ----------
    file : string
        Name of the file to read.

    Returns
    -------
    x, y, z : numpy arrays
        Arrays representing x, y and z coordinates of vertices.
    tri : numpy array
        Triangle definition: each row of tri array represents one triangle
        and the three columns correspond to indices of the three triangle
        vertices.
    '''
    with open(file) as fid:
        mode = 'start'
        x, y, z = list(), list(), list()
        tri = list()

        for line in fid.readlines():
            if line[0] == '#':
                continue

            line = line.replace('\n', '')
            split = line.split(' ')
            if mode == 'start' and len(split) == 3:
                mode = 'coord'
            elif mode == 'coord' and len(split) < 3:
                mode = 'mid'
            elif mode == 'mid' and len(split) == 3:
                mode = 'tri'

            if mode == 'coord' and len(split) == 3:
                [lst.append(float(x)) for x, lst in zip(split, [x, y, z])]
            if mode == 'tri' and len(split) == 3:
                tri.append([int(x) - 1 for x in split])
    tri = np.array(tri, dtype='int')
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return x, y, z, tri


def read_brainnetviewer_nodes(file):
    with open(file) as f:
        xyz, label, size, name = list(), list(), list(), list()
        for line in f.readlines():
            split = line.replace('\n', '').split('\t')
            if len(split) == 6:
                xyz.append([float(x) for x in split[:3]])
                label.append(int(split[3]))
                size.append(float(split[4]))
                name.append(split[-1])
        xyz = np.array(xyz)
        label = np.array(label)
        size = np.array(size)
    return xyz, label, size, name


def read_fibers(file):
    with open(file) as f:
        fibers = list()
        for line in f.readlines():
            splt = line.replace('\n', '').split('   ')
            fibers.append([int(float(x)) for x in splt[1:]])

        return np.array(fibers)


def read_brain(data_dir=None, surface=None, nodes=None):
    if data_dir is None: data_dir = os.getcwd()
    if surface is None: surface = 'BrainMesh_Ch2.nv'
    if nodes is None: nodes = 'Node_AAL90.node'

    surface_file = os.path.join(data_dir, surface)
    x, y, z, tri = read_brainnetviewer_surface(surface_file)
    nodes_file = os.path.join(data_dir, nodes)
    xyz, label, size, name = read_brainnetviewer_nodes(nodes_file)

    brain = dict()
    brain['surface'] = dict(x=x, y=y, z=z, tri=tri)
    brain['nodes'] = dict(xyz=xyz, label=label, size=size, name=name)
    return brain


def plot_brain(brain, surface='dots', nodes=True, node_colors=False,
               node_groups=None, surface_color=None, node_color='red',
               network=None, dot_size=0.1, dot_color='gray', min_fibers=10,
               max_fibers=500, lowest_cmap_color=0.2, highest_cmap_color=0.7,
               cmap='rocket', background='light'):
    import ipyvolume as ipv

    if surface_color is None:
        if surface == 'dots': surface_color = 'gray'
        elif surface == 'full': surface_color = 'orange'

    if isinstance(node_colors, bool) and node_colors:
        node_colors = ['red', 'green', 'blue', 'violet', 'yellow']

    if node_colors and node_groups is None:
        node_groups = brain['nodes']['label']

    fig = ipv.figure()

    # plot surface
    x, y, z = [brain['surface'][key] for key in list('xyz')]
    if surface == 'dots':
        ipv.scatter(x, y, z, marker='box', color=dot_color, size=dot_size)
    elif surface == 'full':
        ipv.plot_trisurf(x, y, z, triangles=brain['surface']['tri'],
                         color=surface_color)

    # plot nodes
    if nodes:
        xyz = brain['nodes']['xyz']
        if node_colors:
            for label_idx, color in zip(np.unique(node_groups), node_colors):
                mask = node_groups == label_idx
                ipv.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2],
                            marker='sphere', color=color, size=1.5)
        else:
             ipv.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='sphere',
                         color=node_color, size=1.5)

    # plot connections
    if network is not None:
        x, y, z = brain['nodes']['xyz'].T
        scaling = highest_cmap_color - lowest_cmap_color
        min_fibers_log = np.log(min_fibers)
        max_fibers_log = np.log(max_fibers)
        if hasattr(plt.cm, cmap):
            cmp = getattr(plt.cm, cmap)
        else:
            import seaborn as sns
            cmp = getattr(sns.cm, cmap)

        with fig.hold_sync():
            for ii in range(89):
                for jj in range(ii, 90):
                    if network[ii, jj] > min_fibers:
                        float_color = (min(np.log((network[ii, jj] - min_fibers))
                                       / (max_fibers_log - min_fibers_log),
                                       1.) * scaling + lowest_cmap_color)
                        line_color = cmp(float_color)
                        ipv.plot(x[[ii, jj]], y[[ii, jj]], z[[ii, jj]],
                                 color=line_color[:3])

    ipv.squarelim()
    ipv.style.use([background, 'minimal'])
    ipv.show()
    return fig
