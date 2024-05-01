import sys, os, datetime, pickle, time, yaml, json
import string, pdb, tqdm
import random, cv2, keras, os.path
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio

from Builder import ConvNetBuilder
from CustomReport import CustomReport

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold

from evidently.metric_preset import ClassificationPreset
from evidently.test_suite import TestSuite
from evidently.tests import *

import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

class Simulator:
    """
    Class to simulate model training and evaluate performance metrics.

    Attributes:
        ws (evidently.ui.workspace.cloud.CloudWorkSpace): Cloud workspace to upload reports and tests.
        project (str): Project on cloud workspace.
        X (numpy.ndarray): Features.
        Y (numpy.ndarray): Labels.
        ass_f (numpy.ndarray): Features for model assessment.
        ass_l (numpy.ndarray): Labels for model assessment.
        verbose (int): Vervosity for prints.

    Methods:
        simulate(NUM_SIMULATIONS, NUM_FOLDERS, results_base_directory): Simulate configurations training and evaluation.
        calculate_statistics(metrics_list): Calculates mean and standard deviation of metrics from a list.
        plot_metrics(mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1, classes): Plots performance metrics.
        plot_confusion_matrix(mean_confusion_matrix, std_dev_confusion_matrix, classes): Plots confusion matrix.
        perform_test_suite(df, tags, metadata, modelo_folder): Performs a Test Suite on data.
        perform_custom_report(model, config_name, df, simulation_date_folder, tags, metadata, fold_no, i): Performs a custom report on data.
    """
    def __init__(self, ws, project, X, Y, ass_f, ass_l, verbose=0):
        self.X = X
        self.Y = Y
        self.ass_f = ass_f
        self.ass_l = ass_l
        self.ws = ws
        self.project_id = project.id
        self.verbose = verbose

        self.SAVE_PERIOD = 1
        self.learning_rate = 2e-4
        self.BATCH_SIZE = 50
        self.STEPS_PER_EPOCH = ((self.X.size)*0.8) / self.BATCH_SIZE
        self.SAVE_PERIOD = 1
        self.epochs = 100
        self.loss = tf.keras.losses.categorical_crossentropy
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.classes = ['OK', 'NOK']
        self.modelPath = os.path.join(os.getcwd(), 'ZN_1D_imgs/bestModel.keras')

        self.checkpoint = ModelCheckpoint( # set model saving checkpoints
                            self.modelPath, # set path to save model weights
                            monitor='loss', # set monitor metrics
                            verbose=1, # set training verbosity
                            save_best_only=True, # set if want to save only best weights
                            save_weights_only=False, # set if you want to save only model weights
                            mode='auto', # set if save min or max in metrics
                            save_freq= int(self.SAVE_PERIOD * self.STEPS_PER_EPOCH) # interval between checkpoints
                            )

        self.earlystopping = EarlyStopping(
                monitor='loss', # set monitor metrics
                min_delta=0.0001, # set minimum metrics delta
                patience=25, # number of epochs to stop training
                restore_best_weights=True, # set if use best weights or last weights
                )

    def simulate(self, NUM_SIMULATIONS, NUM_FOLDERS, results_base_directory):
        """
        Executes model training and evaluation simulations.

        Args:
            num_simulations (int): Number of simulations to run.

        Returns:
            tuple: Tuple containing plots of performance metrics and confusion matrix.
        """
        # Cargar configuraciones desde un archivo YAML
        with open('/home/aacastro/Alejandro/DQ_ACA_2024/A)/configs.yaml') as file:  # CAMBIAR RUTA
            configurations = yaml.safe_load(file)

        # Inicializar diccionario para almacenar métricas de los modelos
        models_metrics = {}

        # Iterar a través de las configuraciones
        for config_name, config_params in configurations['configurations'].items():
            if self.verbose >= 1:
                print(f"Iniciando configuración: '{config_name}'...")
            start_time = time.time()  # Tiempo de inicio para esta configuración

            # Diccionario para almacenar métricas para esta configuración
            config_metrics = {}

            # Iterar a través de las simulaciones
            for i in range(NUM_SIMULATIONS):
                if self.verbose >= 2:
                    print(f"Iniciando simulación {i + 1}...")

                # Crear la carpeta de la fecha de la simulación
                simulation_date_folder = os.path.join(results_base_directory, f"fecha_simulacion_{time.strftime('%Y%m%d')}")
                os.makedirs(simulation_date_folder, exist_ok=True)
                
                # Diccionario para almacenar métricas para cada división en la simulación
                metrics_per_simulation = {"f1_score": [], "recall": [], "precision": [], "roc_auc": [], "confusion_matrix": []}

                # Configurar validación cruzada K-Fold
                kf = KFold(n_splits=NUM_FOLDERS, shuffle=True)

                # Iterar a través de las divisiones
                for fold_no, (train, test) in enumerate(kf.split(self.X, self.Y), 1):
                    if self.verbose >= 2:
                        print(f"Iniciando división {fold_no}...")

                    # Construir el modelo utilizando parámetros de configuración
                    builder = ConvNetBuilder(**config_params)
                    model, tags, metadata = builder.build_model()

                    metadata['num_iteration'] = i + 1
                    metadata['folder'] = fold_no 

                    # Compilar el modelo
                    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
                    model.compile(loss=self.loss, optimizer=optimizer, metrics=['accuracy'])

                    # Entrenar el modelo
                    callbacksList = [self.checkpoint, self.earlystopping]
                    hist = model.fit(self.X[train], self.Y[train], epochs=metadata['epochs'], batch_size=metadata['batch_size'], callbacks=callbacksList)

                    # Evaluar el modelo
                    ypredt = model.predict(self.X[test])
                    y_pred_labels = np.argmax(ypredt, axis=1)
                    pred_prob = ypredt[:, 1]
                    yTestClassT = np.argmax(self.Y[test], axis=1)
                    df_cur = pd.DataFrame({'target': yTestClassT, 'prediction': pred_prob})

                    # Ejecutar un informe de rendimiento de clasificación personalizado
                    modelo_folder, configuration_folder = self.perform_custom_report(model, config_name, df_cur, simulation_date_folder, tags, metadata, fold_no, i)

                    # Calcular métricas adicionales
                    metrics_per_simulation["f1_score"].append(f1_score(yTestClassT, y_pred_labels))
                    metrics_per_simulation["recall"].append(recall_score(yTestClassT, y_pred_labels))
                    metrics_per_simulation["precision"].append(precision_score(yTestClassT, y_pred_labels))
                    metrics_per_simulation["roc_auc"].append(roc_auc_score(yTestClassT, pred_prob))
                    metrics_per_simulation["confusion_matrix"].append(confusion_matrix(yTestClassT, y_pred_labels))

                    # Predict on the independent set.
                    ypredt_ass = model.predict(self.ass_f)
                    pred_prob_ass = ypredt_ass[:, 1]
                    yTestClassT_ass = np.argmax(self.ass_l, axis=1)
                    df_cur_ass = pd.DataFrame({'target':yTestClassT_ass, 'prediction': pred_prob_ass})

                    # Performing Test Suite evaluation with different thresholds
                    self.perform_test_suite(df_cur_ass, tags, metadata, modelo_folder)

                # Calcular la media y la desviación estándar de las métricas para esta simulación
                average_metrics = {}
                for metric, values in metrics_per_simulation.items():
                    mean_metric = np.mean(values, axis=0)
                    std_dev_metric = np.std(values, axis=0)
                    average_metrics[metric] = {"mean": mean_metric, "std_dev": std_dev_metric}

                # Almacenar métricas para esta simulación
                config_metrics[i] = average_metrics
                if self.verbose >= 3:
                    print(f"Config Metrics: {i} {config_metrics[i]}")

            # Calcular la media y la desviación estándar de las métricas para esta configuración
            average_config_metrics = {}
            # Iterar sobre las métricas en el diccionario existente
            for metric in config_metrics[0].keys():
                mean_of_means = np.mean([config_metrics[simulation][metric]['mean'] for simulation in config_metrics.keys()], axis=0)
                mean_of_std_devs = np.mean([config_metrics[simulation][metric]['std_dev'] for simulation in config_metrics.keys()], axis=0)
                
                # Almacenar las medias de las medias y las medias de las desviaciones estándar en el nuevo diccionario
                average_config_metrics[metric] = {'mean': mean_of_means, 'std_dev': mean_of_std_devs}

            # Almacenar métricas para esta configuración
            models_metrics[config_name] = average_config_metrics

            conf_plot = self.plot_confusion_matrix(models_metrics[config_name]['confusion_matrix']['mean'], models_metrics[config_name]['confusion_matrix']['std_dev'], self.classes)
            pio.write_image(conf_plot, os.path.join(configuration_folder, f'confusion_matrix_{config_name}_plot.png'))

            end_time = time.time()  # Tiempo de fin para esta configuración
            elapsed_time = end_time - start_time  # Calcular tiempo transcurrido
            models_metrics[config_name]['execution_time'] = elapsed_time  # Almacenar tiempo transcurrido para esta configuración
            if self.verbose >= 2:
                print(f"Tiempo de ejecución para configuración '{config_name}': {elapsed_time} segundos")

        return models_metrics


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
    
    def perform_test_suite(self, df, tags, metadata, modelo_folder):
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
        thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        for threshold in thresholds:
            binary_classification_performance = TestSuite(tests=[TestPrecisionScore(gte=threshold), 
                                                                    TestRecallScore(gte=threshold), 
                                                                    TestF1Score(gte=threshold), 
                                                                    TestRocAuc(gte=threshold)], 
                                                                    tags=tags, 
                                                                    metadata=metadata)

            binary_classification_performance.run(current_data=df, reference_data=None)
            self.ws.add_test_suite(self.project_id, test_suite=binary_classification_performance)

            # Saving the test suite results in json format
            nombre_archivo_test = f'test_suite_threshold_{threshold}.json'
            ruta_completa_test = os.path.join(modelo_folder, nombre_archivo_test)
            binary_classification_performance.save(ruta_completa_test)
            # dict_test_suite = binary_classification_performance.as_dict()
            # del(dict_test_suite['metrics_preset'])
            # with open(ruta_completa_test, 'w') as file:
            #     json.dump(dict_test_suite, file, indent=4)

    def perform_custom_report(self, model, config_name, df, simulation_date_folder, tags, metadata, fold_no, i):
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
        classification_performance_report = CustomReport(metrics=[ClassificationPreset()], tags=tags, metadata=metadata)
        classification_performance_report.run(reference_data=None, current_data=df)
        self.ws.add_report(self.project_id, classification_performance_report)

        # Guardar el informe en formato JSON
        configuration_folder = os.path.join(simulation_date_folder, f"configuration_{config_name}")
        os.makedirs(configuration_folder, exist_ok=True)
        iteration_folder = os.path.join(configuration_folder, f"iteration_{i+1}")
        os.makedirs(iteration_folder, exist_ok=True)
        modelo_folder = os.path.join(iteration_folder, f"fold_{fold_no}")
        os.makedirs(modelo_folder, exist_ok=True)
        nombre_archivo_rep = f"custom_report.json"
        ruta_completa = os.path.join(modelo_folder, nombre_archivo_rep)
        classification_performance_report.save(ruta_completa)
        # dict_data = classification_performance_report.json()
        # # del(dict_data['metrics_preset'])
        # with open(ruta_completa, 'w') as file:
        #     json.dump(dict_data, file, indent=4)

        save_path_model = os.path.join(modelo_folder, 'model.keras')
        model.save(save_path_model)

        return modelo_folder, configuration_folder
