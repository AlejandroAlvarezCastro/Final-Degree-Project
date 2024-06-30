
import random
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import ops

from sklearn.model_selection import train_test_split

def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")


def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):
        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(ops.concatenate([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
@keras.saving.register_keras_serializable()
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
    return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))


def loss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss

def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


# Función para calcular la similitud utilizando la red siamesa
def calculate_similarity(model, img1, img2):
    """
    Calculates the similarity between two images using a siamese network model.

    Args:
        model (keras.Model): The pre-trained siamese network model.
        img1 (numpy.ndarray): The first image for similarity calculation.
        img2 (numpy.ndarray): The second image for similarity calculation.

    Returns:
        float: The similarity score between img1 and img2.
    """
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    return model.predict([img1, img2])

# Función para clasificar una nueva imagen
def classify_image(model, img, reference_images_class0, reference_images_class1):
    """
    Classifies a single image by comparing it to reference images from two classes.

    Args:
        model (keras.Model): The pre-trained siamese network model.
        img (numpy.ndarray): The image to be classified.
        reference_images_class0 (list of numpy.ndarray): List of reference images for class 0.
        reference_images_class1 (list of numpy.ndarray): List of reference images for class 1.

    Returns:
        int: The predicted class label (0 or 1).
    """
    similarities_class0 = [calculate_similarity(model, img, ref_img) for ref_img in reference_images_class0]
    similarities_class1 = [calculate_similarity(model, img, ref_img) for ref_img in reference_images_class1]
    
    mean_similarity_class0 = np.mean(similarities_class0)
    mean_similarity_class1 = np.mean(similarities_class1)
    
    if mean_similarity_class0 < mean_similarity_class1:
        return 0
    else:
        return 1
    
# Función para clasificar un lote de imágenes
def classify_images(model, images, reference_images_class0, reference_images_class1):
    """
    Classifies a batch of images by comparing each image to reference images from two classes.

    Args:
        model (keras.Model): The pre-trained siamese network model.
        images (list of numpy.ndarray): List of images to be classified.
        reference_images_class0 (list of numpy.ndarray): List of reference images for class 0.
        reference_images_class1 (list of numpy.ndarray): List of reference images for class 1.

    Returns:
        numpy.ndarray: Array of predicted class labels for each image in the batch.
    """
    predictions = []
    for img in images:
        predictions.append(classify_image(model, img, reference_images_class0, reference_images_class1))
    return np.array(predictions)

def prepare_data(X, Y):
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # ass_f = ass_f.astype('float32')
    X_train /= 255
    X_test /= 255
    # ass_f /= 255

    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    # y_train = np.argmax(y_train, axis=1)
    # y_val = np.argmax(y_val, axis=1)
    # Y_test = np.argmax(Y_test, axis=1)

    # Definir máscaras y filtrar x_train utilizándolas
    # mask_ok = y_train == 0
    # mask_nok = y_train == 1
    # x_train_ok = x_train[mask_ok]
    # x_train_nok = x_train[mask_nok]

    pairs_dict = {}
    # make pairs
    pairs_train, labels_train = make_pairs(x_train, y_train)
    x_train_1 = pairs_train[:, 0] 
    x_train_2 = pairs_train[:, 1]
    # x_train_1 = x_train_1.reshape(x_train_1.shape[0], -1)
    # x_train_2 = x_train_2.reshape(x_train_2.shape[0], -1)
    pairs_dict['train'] = {'data': [x_train_1, x_train_2], 'labels': labels_train}

    pairs_val, labels_val = make_pairs(x_val, y_val)
    x_val_1 = pairs_val[:, 0]  
    x_val_2 = pairs_val[:, 1]
    # x_val_1 = x_val_1.reshape(x_val_1.shape[0], -1)
    # x_val_2 = x_val_2.reshape(x_val_2.shape[0], -1)
    pairs_dict['val'] = {'data': [x_val_1, x_val_2], 'labels': labels_val}

    pairs_test, labels_test = make_pairs(X_test, Y_test)
    x_test_1 = pairs_test[:, 0]
    x_test_2 = pairs_test[:, 1]
    # x_test_1 = x_test_1.reshape(x_test_1.shape[0], -1)
    # x_test_2 = x_test_2.reshape(x_test_2.shape[0], -1)
    pairs_dict['test'] = {'data': [x_test_1, x_test_2], 'labels': labels_test}

    return pairs_dict
