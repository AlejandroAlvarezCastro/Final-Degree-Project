import sys, os, datetime, pickle, time
import string, pdb, tqdm
import random, cv2, keras, os.path
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class Simulator:
    """
    Class to simulate model training and evaluate performance metrics.

    Attributes:
        yaml_file (str): Path to the YAML file containing model configurations.
        model_name (str): Name of the model.
        model (keras.Model): Keras model to be trained and evaluated.
        train_f (numpy.ndarray): Training features.
        train_l (numpy.ndarray): Training labels.
        valid_f (numpy.ndarray): Validation features.
        valid_l (numpy.ndarray): Validation labels.
        ass_f (numpy.ndarray): Features for model assessment.
        ass_l (numpy.ndarray): Labels for model assessment.

    Methods:
        run_simulations(num_simulations): Executes model training and evaluation simulations.
        calculate_statistics(metrics_list): Calculates mean and standard deviation of metrics from a list.
        plot_metrics(mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1, classes): Plots performance metrics.
        plot_confusion_matrix(mean_confusion_matrix, std_dev_confusion_matrix, classes): Plots confusion matrix.
    """
    def __init__(self, yaml_file, model_name, model, train_f, train_l, valid_f, valid_l, ass_f, ass_l):
        self.yaml_file = yaml_file
        self.model_name = model_name
        self.model = model
        self.train_f = train_f
        self.train_l = train_l
        self.valid_f = valid_f
        self.valid_l = valid_l
        self.ass_f = ass_f
        self.ass_l = ass_l

        self.learning_rate = 2e-4
        self.BATCH_SIZE = 50
        self.STEPS_PER_EPOCH = train_l.size / self.BATCH_SIZE
        self.SAVE_PERIOD = 1
        self.epochs = 100
        self.loss = tf.keras.losses.categorical_crossentropy
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.classes = ['OK', 'NOK']

    def run_simulations(self, num_simulations):
        """
        Executes model training and evaluation simulations.

        Args:
            num_simulations (int): Number of simulations to run.

        Returns:
            tuple: Tuple containing plots of performance metrics and confusion matrix.
        """
        precision_scores = []
        recall_scores = []
        f1_scores = []
        accuracy_scores = []
        confusion_matrices = []

        resultados_dir = os.path.join(os.getcwd(), 'resultados')
        yaml_name = os.path.splitext(os.path.basename(self.yaml_file))[0]
        yaml_dir = os.path.join(resultados_dir, yaml_name)
        model_dir = os.path.join(yaml_dir, self.model_name)

        os.makedirs(resultados_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        save_path = os.path.join(os.getcwd(), 'ZN_1D_imgs/')
        modelPath = os.path.join(os.getcwd(), 'ZN_1D_imgs/bestModel.keras')

        for _ in range(num_simulations):
            model = self.model

            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

            checkpoint = keras.callbacks.ModelCheckpoint(
                modelPath,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                save_freq=int(self.SAVE_PERIOD * self.STEPS_PER_EPOCH)
            )

            earlystopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=25,
                restore_best_weights=True
            )

            callbacksList = [checkpoint, earlystopping]

            hist = model.fit(self.train_f, self.train_l, epochs=self.epochs, batch_size=self.BATCH_SIZE,
                             validation_data=(self.valid_f, self.valid_l), callbacks=callbacksList)

            with open(os.path.join(save_path, f"hist_{_}.pkl"), "wb") as file:
                pickle.dump(hist.history, file)

            yPredClass = np.argmax(model.predict(self.ass_f), axis=-1)
            yTestClass = np.argmax(self.ass_l, axis=1)

            precision = precision_score(yTestClass, yPredClass, average=None)
            recall = recall_score(yTestClass, yPredClass, average=None)
            f1 = f1_score(yTestClass, yPredClass, average=None)
            accuracy = accuracy_score(yTestClass, yPredClass)
            confusion_matrix_sim = confusion_matrix(yTestClass, yPredClass)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)
            confusion_matrices.append(confusion_matrix_sim)

        precision_scores = np.array(precision_scores)
        recall_scores = np.array(recall_scores)
        f1_scores = np.array(f1_scores)
        accuracy_scores = np.array(accuracy_scores)
        confusion_matrices = np.array(confusion_matrices)

        # Calculate statistics
        mean_precision, std_dev_precision = self.calculate_statistics(precision_scores)
        mean_recall, std_dev_recall = self.calculate_statistics(recall_scores)
        mean_f1, std_dev_f1 = self.calculate_statistics(f1_scores)
        mean_accuracy, std_dev_accuracy = self.calculate_statistics(accuracy_scores)
        mean_confusion_matrix, std_dev_confusion_matrix = self.calculate_statistics(confusion_matrices)


        model_summary_img_path = os.path.join(model_dir, "model_summary.png")
        plot_model(model, to_file=model_summary_img_path, show_shapes=True, show_layer_names=True)

        metrics_plot = self.plot_metrics(mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1, self.classes)
        conf_plot = self.plot_confusion_matrix(mean_confusion_matrix, std_dev_confusion_matrix, self.classes)

        pio.write_image(metrics_plot, os.path.join(model_dir, f'metrics_{str(self.yaml_file)}_{str(self.model_name)}_plot.png'))
        pio.write_image(conf_plot, os.path.join(model_dir, f'confusion_matrix_{str(self.yaml_file)}_{str(self.model_name)}_plot.png'))

        return metrics_plot, conf_plot

    def calculate_statistics(self, metrics_list):
        """
        Calculates the mean and standard deviation of a list of metrics.

        Args:
            metrics_list (list): List of metrics.

        Returns:
            tuple: Mean and standard deviation of the metrics.
        """
        metrics_array = np.array(metrics_list)
        mean_metrics = np.mean(metrics_array, axis=0)
        std_dev_metrics = np.std(metrics_array, axis=0)
        return mean_metrics, std_dev_metrics

    def plot_metrics(self, mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1, classes):
        """
        Plots performance metrics.

        Args:
            mean_precision (numpy.ndarray): Mean precision values.
            std_dev_precision (numpy.ndarray): Standard deviation of precision values.
            mean_recall (numpy.ndarray): Mean recall values.
            std_dev_recall (numpy.ndarray): Standard deviation of recall values.
            mean_f1 (numpy.ndarray): Mean F1-score values.
            std_dev_f1 (numpy.ndarray): Standard deviation of F1-score values.
            classes (list): List of class labels.

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        metrics = {
            'Precision': {'mean': mean_precision, 'stddev': std_dev_precision, 'color': 'rgb(153, 43, 132)'},
            'Recall': {'mean': mean_recall, 'stddev': std_dev_recall, 'color': 'rgb(255, 153, 51)'},
            'F1-Score': {'mean': mean_f1, 'stddev': std_dev_f1, 'color': 'rgb(51, 153, 255)'}}

        fig = go.Figure()

        for metric_name, metric_data in metrics.items():
            fig.add_trace(go.Bar(
                x=classes,
                y=metric_data['mean'],
                error_y=dict(type='data', array=metric_data['stddev'], visible=True),
                name=metric_name,
                marker_color=metric_data['color']
            ))

        fig.update_layout(
            title='Metrics per class',
            xaxis=dict(title='Classes'),
            yaxis=dict(title='Value'),
            yaxis_tickformat=".2%",
            barmode='group',
            bargap=0.15)

        return fig

    def plot_confusion_matrix(self, mean_confusion_matrix, std_dev_confusion_matrix, classes):
        """
        Plots the confusion matrix.

        Args:
            mean_confusion_matrix (numpy.ndarray): Mean confusion matrix values.
            std_dev_confusion_matrix (numpy.ndarray): Standard deviation of confusion matrix values.
            classes (list): List of class labels.

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """
        fig = go.Figure(data=go.Heatmap(
            z=mean_confusion_matrix,
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

        fig.update_layout(
            title='Confusion Matrix (Mean +/- Standard Deviation)',
            xaxis=dict(tickvals=list(range(len(classes))), ticktext=classes),
            yaxis=dict(tickvals=list(range(len(classes))), ticktext=classes),
            annotations=annotations)

        return fig
