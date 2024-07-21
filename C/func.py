import keras 
import tensorflow as tf
from keras import ops
import numpy as np
import random
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
# from evidently.metrics import classification_performance
from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.metric_preset import ClassificationPreset
# from evidently.metric_preset import MetricPreset
from evidently.test_suite import *
from evidently.tests import *
from evidently.report import Report
from evidently.metrics import ClassificationQualityMetric, ClassificationConfusionMatrix, ClassificationQualityByClass
# from evidently import Options


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


# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

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
    pred = model.predict([img1, img2])
    print("Pred:", pred)
    # pred = model.predict([img1, img2])[0][0]
    return pred

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
    
    # mean_similarity_class0 = np.mean(similarities_class0)
    # mean_similarity_class1 = np.mean(similarities_class1)
    # Encontrar la mayor similitud individual para cada clase
    # print("similarities 0: ", similarities_class0)
    # print("similarities 1: ", similarities_class1)
    # print("type: ", type(similarities_class0))
    min_similarity_class0 = min(similarities_class0)
    min_similarity_class1 = min(similarities_class1)
    # print(f'MIn class 0 {min_similarity_class0}, MIn class 1 {min_similarity_class1}')
    
    # random = 0
    if min_similarity_class0 < min_similarity_class1:
        # print("Añado 0")
        return 0
    elif min_similarity_class1 < min_similarity_class0:
        # print("Añado 1")
        return 1
    else:
        clas =  random.choices([0, 1], weights=[60, 40], k=1)[0]
        # print("Añado ", clas)
        return clas
# def classify_image(model, img, reference_images_class0, reference_images_class1):
#     """
#     Classifies a single image by comparing it to reference images from two classes.

#     Args:
#         model (keras.Model): The pre-trained siamese network model.
#         img (numpy.ndarray): The image to be classified.
#         reference_images_class0 (list of numpy.ndarray): List of reference images for class 0.
#         reference_images_class1 (list of numpy.ndarray): List of reference images for class 1.

#     Returns:
#         tuple: The predicted class label (0 or 1) and the array of probabilities.
#     """
#     similarities_class0 = [calculate_similarity(model, img, ref_img) for ref_img in reference_images_class0]
#     similarities_class1 = [calculate_similarity(model, img, ref_img) for ref_img in reference_images_class1]
    
#     # Calcula la mayor similitud individual para cada clase
#     min_similarity_class0 = min(similarities_class0)
#     min_similarity_class1 = min(similarities_class1)

#     # print("Similarities 0: ", similarities_class0)
#     # print("Similarities 1: ", similarities_class1)
#     # print(f'Min class 0 {min_similarity_class0}, Min class 1 {min_similarity_class1}')

#     # Convertir las similitudes en probabilidades (inversas porque 0 es más similar)
#     similarities = np.array([min_similarity_class0, min_similarity_class1])
#     inverse_similarities = 1 - similarities
#     probabilities = softmax(inverse_similarities)
#     # print("Probabilidades: ", probabilities)
    
#     # Clasificar basado en las probabilidades
#     clas = np.random.choice([0, 1], p=probabilities)
#     print("Añado ", clas)
#     return clas, probabilities

# Función para clasificar un lote de imágenes
# def classify_images(model, images, reference_images_class0, reference_images_class1):
#     """
#     Classifies a batch of images by comparing each image to reference images from two classes.

#     Args:
#         model (keras.Model): The pre-trained siamese network model.
#         images (list of numpy.ndarray): List of images to be classified.
#         reference_images_class0 (list of numpy.ndarray): List of reference images for class 0.
#         reference_images_class1 (list of numpy.ndarray): List of reference images for class 1.

#     Returns:
#         tuple: Array of predicted class labels and array of probability arrays for each image in the batch.
#     """
#     predictions = []
#     probabilities_list = []
#     for img in images:
#         clas, probabilities = classify_image(model, img, reference_images_class0, reference_images_class1)
#         predictions.append(clas)
#         probabilities_list.append(probabilities)
#     return np.array(predictions), np.array(probabilities_list)
        
    
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
    plt.ylim(0, 1)
    plt.show()

def conf_mat(ass_l, predicted_labels):
    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(ass_l, predicted_labels), annot=True, fmt="d", cmap="Blues", xticklabels=['OK', 'NOK'], yticklabels=['OK', 'NOK'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            # Recursively reset weights of sub-models
            reset_weights(layer)
            continue
        
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))
        
        if isinstance(layer, keras.layers.BatchNormalization):
            # Reset moving mean and variance for BatchNormalization layers
            for attr in ['moving_mean', 'moving_variance']:
                obj = getattr(layer, attr, None)
                if obj is not None:
                    obj.assign(tf.zeros_like(obj))

def store_simulation_data(detailed_metrics, i, fold_no, train, test, yTestClassT, y_pred_labels):
    # Almacenar los índices de las muestras de entrenamiento y prueba en el diccionario
    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["training_indexes"] = train
    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["test_indexes"] = test
    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["predicted_labels"] = y_pred_labels

    # Calcular métricas adicionales
    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["f1_score"].append(f1_score(yTestClassT, y_pred_labels))
    print("F1 en este mdoelo: ", f1_score(yTestClassT, y_pred_labels))
    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["recall"].append(recall_score(yTestClassT, y_pred_labels))
    print("Recall en este mdoelo: ", recall_score(yTestClassT, y_pred_labels))
    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["precision"].append(precision_score(yTestClassT, y_pred_labels))
    print("Precision en este mdoelo: ", precision_score(yTestClassT, y_pred_labels))
    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["roc_auc"].append(roc_auc_score(yTestClassT, y_pred_labels))
    print("ROCAUC en este mdoelo: ", roc_auc_score(yTestClassT, y_pred_labels))
    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["confusion_matrix"].append(confusion_matrix(yTestClassT, y_pred_labels))


def generate_empty_dict(NUM_FOLDERS, NUM_SIMULATIONS):
    # Inicializar el diccionario para almacenar las métricas detalladamente
    detailed_metrics = {}
    for sim in range(1, NUM_SIMULATIONS + 1):
        detailed_metrics[f'sim_{sim}'] = {}

        for fold in range(1, NUM_FOLDERS + 1):
            detailed_metrics[f'sim_{sim}'][f'fold_{fold}'] = {
                'metrics': {
                    'f1_score': [],
                    'recall': [],
                    'precision': [],
                    'roc_auc': [],
                    'confusion_matrix': []
                },
                'training_indexes': None,
                'test_indexes': None,
                'predicted_labels': None,
                'predicted_probabilities': None}
            
    return detailed_metrics

# def find_best_f1_score(detailed_metrics):
#     """
#     Encuentra las claves de simulación y fold que tienen el mayor f1_score válido,
#     y devuelve el campo 'predicted_labels' asociado.

#     Args:
#         detailed_metrics (dict): Diccionario con las métricas detalladas.

#     Returns:
#         tuple: Claves de la simulación y el fold con el mayor f1_score válido y los 'predicted_labels'.
#     """
#     best_f1_score = -1
#     best_keys = (None, None)
#     best_predicted_labels = None

#     for sim_key, sim_value in detailed_metrics.items():
#         for fold_key, fold_value in sim_value.items():
#             f1_scores = fold_value['metrics'].get('f1_score', [])
#             if f1_scores:  # Asegurarse de que la lista no esté vacía
#                 max_f1_score = max(f1_scores)
#                 if max_f1_score > best_f1_score and not (max_f1_score == 0.6666666666666666 or max_f1_score == 0):
#                     best_f1_score = max_f1_score
#                     best_keys = (sim_key, fold_key)
#                     best_predicted_labels = fold_value['predicted_labels']

#     return best_keys, best_f1_score, best_predicted_labels

# def find_best_f1_score(detailed_metrics):
def find_best_f1_score(detailed_metrics):
    """
    Encuentra las claves de simulación y fold que tienen el mayor f1_score válido,
    y devuelve el campo 'predicted_labels' asociado, ignorando la clave 'best_model'.

    Args:
        detailed_metrics (dict): Diccionario con las métricas detalladas.

    Returns:
        tuple: Claves de la simulación y el fold con el mayor f1_score válido y los 'predicted_labels'.
    """
    import numpy as np

    best_f1_score = -1
    best_keys = (None, None)
    best_predicted_labels = None

    for sim_key, sim_value in detailed_metrics.items():
        if sim_key == 'best_model':  # Ignorar la clave 'best_model'
            continue
        for fold_key, fold_value in sim_value.items():
            f1_scores = fold_value['metrics'].get('f1_score', [])
            if f1_scores:  # Asegurarse de que la lista no esté vacía
                max_f1_score = max(f1_scores)
                if max_f1_score > best_f1_score and not (max_f1_score == 0.66666666 or max_f1_score == 0):
                    best_f1_score = max_f1_score
                    best_keys = (sim_key, fold_key)
                    best_predicted_labels = fold_value['predicted_labels']

    # Si el mejor f1_score es 0.66666666 o 0, ajustamos los valores según lo solicitado
    if best_f1_score == 0.66666666 or best_f1_score == 0:
        best_f1_score = 0
        best_keys = ('sim1', 'fold1')
        best_predicted_labels = np.ones(60)

    return best_keys, best_f1_score, best_predicted_labels

    """
    Encuentra las claves de simulación y fold que tienen el mayor f1_score válido,
    y devuelve el campo 'predicted_labels' asociado, ignorando la clave 'best_model'.

    Args:
        detailed_metrics (dict): Diccionario con las métricas detalladas.

    Returns:
        tuple: Claves de la simulación y el fold con el mayor f1_score válido y los 'predicted_labels'.
    """
    best_f1_score = -1
    best_keys = (None, None)
    best_predicted_labels = None

    for sim_key, sim_value in detailed_metrics.items():
        if sim_key == 'best_model':  # Ignorar la clave 'best_model'
            continue
        for fold_key, fold_value in sim_value.items():
            f1_scores = fold_value['metrics'].get('f1_score', [])
            if f1_scores:  # Asegurarse de que la lista no esté vacía
                max_f1_score = max(f1_scores)
                if max_f1_score > best_f1_score and not (max_f1_score == 0.6666666666666666 or max_f1_score == 0):
                    best_f1_score = max_f1_score
                    best_keys = (sim_key, fold_key)
                    best_predicted_labels = fold_value['predicted_labels']
                # else:
                #     best_predicted_labels = np.zeros(1)

    return best_keys, best_f1_score, best_predicted_labels

# class BinaryClassificationPreset(MetricPreset):
#     def __init__(self):
#         super().__init__(metrics=[
#             ClassificationQualityMetric(),
#             ClassificationConfusionMatrix(),
#             ClassificationQualityByClass()
#         ])

def perform_report(ws, project_id, configuration_folder, df, tags, metadata):
    """
    Performs a custom classification performance report and saves it in JSON format.

    Args:
        model (keras.Model): Trained model.
        config_name (str): Name of the configuration.
        df (pandas.DataFrame): Dataframe containing target and prediction columns.
        simulation_date_folder (str): Path to the folder for the simulation date.
        tags (list): List of tags associated with the report.
        metadata (dict): Metadata associated with the report.
        fold_no (int): Fold number.
        i (int): Iteration number.

    Returns:
        tuple: Paths to the folders where the custom report and model are saved.
    """
    # Crear opciones para desactivar métricas probabilísticas
    # options = AnyOptions(include_roc_auc=False, include_log_loss=False)
    # classification_performance_report = Report(metrics=[ClassificationPreset()], tags=tags, metadata=metadata)
    classification_performance_report = Report(metrics=[
            # ClassificationQualityMetric(),
            ClassificationConfusionMatrix(),
            ClassificationQualityByClass()
        ], tags=tags, metadata=metadata)
    classification_performance_report.run(reference_data=None, current_data=df)
    ws.add_report(project_id, classification_performance_report)

    nombre_archivo_rep = f"custom_report.json"
    ruta_completa = os.path.join(configuration_folder, nombre_archivo_rep)
    classification_performance_report.save(ruta_completa)

def perform_test_suite(ws, project_id, df, tags, metadata, configuration_folder, thresholds):
    """
    Performs evaluation using a test suite for different thresholds.

    Args:
        df (pandas.DataFrame): Dataframe containing target and prediction columns.
        tags (list): List of tags associated with the evaluation.
        metadata (dict): Metadata associated with the evaluation.
        modelo_folder (str): Path to the folder to save test suite results.

    Returns:
        None
    """
    # thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    for threshold in thresholds:
        binary_classification_performance = TestSuite(tests=[TestPrecisionScore(gte=threshold), 
                                                                TestRecallScore(gte=threshold), 
                                                                TestF1Score(gte=threshold), #TestRocAuc(gte=threshold)
                                                                    ], 
                                                                tags=tags, 
                                                                metadata=metadata)

        binary_classification_performance.run(current_data=df, reference_data=None)
        ws.add_test_suite(project_id, test_suite=binary_classification_performance)

        # Saving the test suite results in json format
        ruta_completa_test = os.path.join(configuration_folder, f'test_suite_threshold_{threshold}.json')
        binary_classification_performance.save(ruta_completa_test)  

def SetUp_CloudWorkSpace(path):
    # Cargar configuraciones desde un archivo YAML
    with open(path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    
    evidently_config = yaml_content.get('evidently', {})
    token = evidently_config.get('token', None)
    url = evidently_config.get('url', None)
    project_id = evidently_config.get('project_id', None)

    ws = CloudWorkspace(token=token, url=url)

    return ws, project_id