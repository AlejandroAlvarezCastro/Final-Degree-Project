import sys, os, datetime, pickle, time
import string, pdb, tqdm
import random, cv2, keras, os.path
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio

import evidently
from evidently.options import ColorOptions
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
# from evidently.dashboard import Dashboard
# from evidently.dashboard.tabs import ProbClassificationPerformanceTab
# from evidently.model_profile import Profile
# from evidently.profile_sections import ClassificationPerformanceProfileSection
# from evidently.tabs import ClassificationPerformanceTab

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def simulations(model_name, num_simulations, model, train_f, train_l, valid_f, valid_l, ass_f, ass_l):
    """
    Performs simulations of neural network training and calculates performance metrics.

    Args:
        model_name (str): Name of the model.
        num_simulations (int): Number of simulations to perform.
        model (tf.keras.Model): Neural network model to train.
        train_f (numpy.ndarray): Training features.
        train_l (numpy.ndarray): Training labels.
        valid_f (numpy.ndarray): Validation features.
        valid_l (numpy.ndarray): Validation labels.
        ass_f (numpy.ndarray): Features of the independent test set.
        ass_l (numpy.ndarray): Labels of the independent test set.

    Returns:
        tuple: A tuple containing two plots and performance metrics.
            - metrics_plot: Metrics plot.
            - conf_plot: Confusion matrix plot.
    """
    learning_rate   = 2e-4
    BATCH_SIZE      = 50
    STEPS_PER_EPOCH = train_l.size / BATCH_SIZE
    SAVE_PERIOD     = 1
    epochs = 100

    #
    loss = tf.keras.losses.categorical_crossentropy
    # loss = tf.keras.losses.binary_crossentropy

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    classes = ['OK', 'NOK']
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    confusion_matrices = []

    # Construct directory name
    resultados_dir = os.path.join(os.getcwd(), 'resultados')
    model_dir = os.path.join(resultados_dir, model_name)
    # Create "resultados" directory if it doesn't exist
    os.makedirs(resultados_dir, exist_ok=True)
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(os.getcwd(), 'ZN_1D_imgs/')
    modelPath = os.path.join(os.getcwd(), 'ZN_1D_imgs/bestModel.keras')

    # Perform 10 simulations of network training
    for _ in range(num_simulations):
        # Create the model
        model = model

        # Compile the model
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # Configure checkpoints
        checkpoint = keras.callbacks.ModelCheckpoint(
            modelPath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq=int(SAVE_PERIOD * STEPS_PER_EPOCH)
        )

        earlystopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=25,
            restore_best_weights=True
        )

        callbacksList = [checkpoint, earlystopping]

        # Train the model
        hist = model.fit(train_f, train_l, epochs=epochs, batch_size=BATCH_SIZE,
                        validation_data=(valid_f, valid_l), callbacks=callbacksList) 

        # Save training history
        with open(os.path.join(save_path, f"hist_{_}.pkl"), "wb") as file:
            pickle.dump(hist.history, file)

        # Get predictions on the independent set
        yPredClass = np.argmax(model.predict(ass_f), axis=-1)
        yTestClass = np.argmax(ass_l, axis=1)

        # Run evidently AI test
        # run_evidently_test(yPredClass, yTestClass)

        # Calculate metrics per class
        precision = precision_score(yTestClass, yPredClass, average=None)
        recall = recall_score(yTestClass, yPredClass, average=None)
        f1 = f1_score(yTestClass, yPredClass, average=None)
        accuracy = accuracy_score(yTestClass, yPredClass)
        confusion_matrix_sim = confusion_matrix(yTestClass, yPredClass)

        # Store metrics for this simulation
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        confusion_matrices.append(confusion_matrix_sim)

    # Convert lists to arrays to facilitate calculation of mean and standard deviation
    precision_scores = np.array(precision_scores)
    recall_scores = np.array(recall_scores)
    f1_scores = np.array(f1_scores)
    accuracy_scores = np.array(accuracy_scores)
    confusion_matrices = np.array(confusion_matrices)

    # Calculate mean and standard deviation of metrics per class
    mean_precision = np.mean(precision_scores, axis=0)
    mean_recall = np.mean(recall_scores, axis=0)
    mean_f1 = np.mean(f1_scores, axis=0)
    mean_accuracy = np.mean(accuracy_scores)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

    std_dev_precision = np.std(precision_scores, axis=0)
    std_dev_recall = np.std(recall_scores, axis=0)
    std_dev_f1 = np.std(f1_scores, axis=0)
    std_dev_accuracy = np.std(accuracy_scores)
    std_dev_confusion_matrix = np.std(confusion_matrices, axis=0)

    # Print results
    print("\nMean Precision:", mean_precision)
    print("Mean Recall:", mean_recall)
    print("Mean F1-score:", mean_f1)
    print("Mean Accuracy:", mean_accuracy)
    print("Mean Conf. Matrix", mean_confusion_matrix)

    print("\nStandard Deviation of Precision:", std_dev_precision)
    print("Standard Deviation of Recall:", std_dev_recall)
    print("Standard Deviation of F1-score:", std_dev_f1)
    print("Standard Deviation of Accuracy:", std_dev_accuracy)
    print("Standart Conf. Matrix", std_dev_confusion_matrix)

    # Save model summary as an image
    model_summary_img_path = os.path.join(model_dir, "model_summary.png")
    plot_model(model, to_file=model_summary_img_path, show_shapes=True, show_layer_names=True)

    # Obtain a plot to visualize results
    metrics_plot = plot_metrics(mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1, classes)

    conf_plot = plot_confusion_matrix(mean_confusion_matrix, std_dev_confusion_matrix, classes)

    # Save the plots in the specified directory
    pio.write_image(metrics_plot, os.path.join(model_dir, 'metrics_plot.png'))
    pio.write_image(conf_plot, os.path.join(model_dir, 'confusion_matrix_plot.png'))

    # return metrics_plot, conf_plot, mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1, mean_accuracy, std_dev_accuracy
    return metrics_plot, conf_plot

def plot_metrics(mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1, classes):
    """
    Plots metrics such as Precision, Recall, and F1-Score per class.

    Args:
        mean_precision (numpy.ndarray): Mean precision values per class.
        std_dev_precision (numpy.ndarray): Standard deviation of precision values per class.
        mean_recall (numpy.ndarray): Mean recall values per class.
        std_dev_recall (numpy.ndarray): Standard deviation of recall values per class.
        mean_f1 (numpy.ndarray): Mean F1-Score values per class.
        std_dev_f1 (numpy.ndarray): Standard deviation of F1-Score values per class.
        classes (list): List of class labels.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object containing the metrics plot.
    """
    metrics = {
        'Precision': {
            'mean': mean_precision,
            'stddev': std_dev_precision,
            'color': 'rgb(153, 43, 132)'},
        'Recall': {
            'mean': mean_recall,
            'stddev': std_dev_recall,
            'color': 'rgb(255, 153, 51)'},
        'F1-Score': {
            'mean': mean_f1,
            'stddev': std_dev_f1,
            'color': 'rgb(51, 153, 255)'}}

    # Create the figure
    fig = go.Figure()

    # Add bars for each metric
    for metric_name, metric_data in metrics.items():
        fig.add_trace(go.Bar(
            x=classes,
            y=metric_data['mean'],
            error_y=dict(type='data', array=metric_data['stddev'], visible=True),
            name=metric_name,
            marker_color=metric_data['color']))

    # Update the layout of the plot
    fig.update_layout(
        title='Metrics per class',
        xaxis=dict(title='Classes'),
        yaxis=dict(title='Value'),
        yaxis_tickformat=".2%",
        barmode='group',
        bargap=0.15)

    return fig


def plot_confusion_matrix(mean_confusion_matrix, std_dev_confusion_matrix, classes):
    """
    Plots the confusion matrix with mean values and standard deviations.

    Args:
        mean_confusion_matrix (numpy.ndarray): Mean values of the confusion matrix.
        std_dev_confusion_matrix (numpy.ndarray): Standard deviations of the confusion matrix.
        classes (list): List of class labels.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object containing the confusion matrix plot.
    """
    fig = go.Figure(data=go.Heatmap(z=mean_confusion_matrix,
                                     zmin=mean_confusion_matrix.min(),
                                     zmax=mean_confusion_matrix.max(),
                                     hoverongaps=False,
                                     colorscale='Blues',
                                     colorbar=dict(title='Count')))

    annotations = []

    for i in range(mean_confusion_matrix.shape[0]):
        for j in range(mean_confusion_matrix.shape[1]):
            annotation = {
                'x': j,
                'y': i,
                'text': f"{mean_confusion_matrix[i, j]:.2f} ± {std_dev_confusion_matrix[i, j]:.2f}",
                'showarrow': False,
                'font': {'color': 'black'}
            }
            annotations.append(annotation)

    fig.update_layout(title='Confusion Matrix (Mean +/- Standard Deviation)',
                      xaxis=dict(tickvals=list(range(len(classes))), ticktext=classes),
                      yaxis=dict(tickvals=list(range(len(classes))), ticktext=classes),
                      annotations=annotations)
    return fig


def run_evidently_test(predictions_array, y_true):
    # Create a Profile

    curr_data = {
        'target': y_true,
        'predictions': predictions_array}

    curr_data = pd.DataFrame(curr_data)
    
    classification_performance_dataset_tests =TestSuite(tests=[
                                                        TestAccuracyScore(),
                                                        TestPrecisionScore(),
                                                        TestRecallScore(),
                                                        TestF1Score(),
                                                        TestPrecisionByClass(label=0),
                                                        TestPrecisionByClass(label=1),
                                                        TestPrecisionByClass(label=2),
                                                        TestRecallByClass(label=0),
                                                        TestRecallByClass(label=1),
                                                        TestRecallByClass(label=2),
                                                        TestF1ByClass(label=0),
                                                        TestF1ByClass(label=1),
                                                        TestF1ByClass(label=2)])
    
    # Run the test
    classification_performance_dataset_tests.run(reference_data=None, current_data=curr_data)

    # Upload report to EvidentlyAI
    ws = CloudWorkspace(
        token="dG9rbgF5sQ3u67hNVo7e7sL4fnJ4+9DpuR+b778AZBtKinzUygBQuzpCQ5CLgwI9Gu+bIOgKn7c7e8SscZauU2bz70HJuk6cPV14XyrDKKl3Tg1SXt5FsEIp9bSTX5sOYXqXABHlDJlbcjobYNq4dZ2Wa6dhspPZQVb9",
        url="https://app.evidently.cloud"
    )

    project = ws.get_project("3fb185be-7c79-494b-9401-d789a877c4ca")

    ws.add_test_suite(project.id, classification_performance_dataset_tests)


