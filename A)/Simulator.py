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

from evidently.ui.workspace.cloud import CloudWorkspace
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
        yaml_path (str): Path to the yaml with the information.
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
        SetUp_CloudWorkSpace(path): Sets up the cloud working space environment
    """
    def __init__(self, yaml_path, X, Y, ass_f, ass_l, verbose=0):
        self.X = X
        self.Y = Y
        self.ass_f = ass_f
        self.ass_l = ass_l
        self.yaml_path = yaml_path
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
        
        # Setting up the cloud working space environment
        self.ws, self.project_id = self.SetUp_CloudWorkSpace(yaml_path)
        

    def simulate(self, NUM_SIMULATIONS, NUM_FOLDERS, results_base_directory):
        """
        Executes model training and evaluation simulations.

        Args:
            num_simulations (int): Number of simulations to run.

        Returns:
            tuple: Tuple containing plots of performance metrics and confusion matrix.
        """
        # Cargar configuraciones desde un archivo YAML
        yaml_content = self.load_yaml(self.yaml_path)

        models = self.build_models(yaml_content['configurations'], yaml_content['architectures'])

        # Inicializar diccionario para almacenar métricas de los modelos
        models_metrics = {}

        # Iterar sobre los modelos construidos
        for k in range(len(models)):
            model_name, model, tags, metadata, thresholds = models[k]
            if self.verbose >= 1:
                print(f"Iniciando modelo: '{model_name}'...")
            start_time = time.time()  # Tiempo de inicio para esta configuración

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
                        'test_indexes': None}

            print("Detailed metrics vacio: ", detailed_metrics)

            # Iterar a través de las simulaciones
            for i in range(NUM_SIMULATIONS):
                if self.verbose >= 2:
                    print(f"Iniciando simulación {i + 1}...")

                # Crear la carpeta de la fecha de la simulación
                simulation_date_folder = os.path.join(results_base_directory, f"fecha_simulacion_{time.strftime('%Y%m%d')}")
                os.makedirs(simulation_date_folder, exist_ok=True)

                # Configurar validación cruzada K-Fold
                kf = KFold(n_splits=NUM_FOLDERS, shuffle=True)

                # Iterar a través de las divisiones
                for fold_no, (train, test) in enumerate(kf.split(self.X, self.Y), 1):
                    if self.verbose >= 2:
                        print(f"Iniciando división {fold_no}...")
                    
                    metadata = metadata
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
                    modelo_folder, configuration_folder = self.perform_custom_report(model, model_name, df_cur, simulation_date_folder, tags, metadata, fold_no, i)

                    # Almacenar los índices de las muestras de entrenamiento y prueba en el diccionario
                    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["training_indexes"] = train
                    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["test_indexes"] = test

                    # Calcular métricas adicionales
                    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["f1_score"].append(f1_score(yTestClassT, y_pred_labels))
                    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["recall"].append(recall_score(yTestClassT, y_pred_labels))
                    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["precision"].append(precision_score(yTestClassT, y_pred_labels))
                    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["roc_auc"].append(roc_auc_score(yTestClassT, pred_prob))
                    detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["confusion_matrix"].append(confusion_matrix(yTestClassT, y_pred_labels))

                    # Predict on the independent set.
                    ypredt_ass = model.predict(self.ass_f)
                    pred_prob_ass = ypredt_ass[:, 1]
                    yTestClassT_ass = np.argmax(self.ass_l, axis=1)
                    df_cur_ass = pd.DataFrame({'target':yTestClassT_ass, 'prediction': pred_prob_ass})

                    # Performing Test Suite evaluation with different thresholds
                    self.perform_test_suite(df_cur_ass, tags, metadata, modelo_folder, thresholds)

            if self.verbose >= 3:
                print(f"Detailed metrics for config {model_name}: ", detailed_metrics)

            path_detailed_metrics = os.path.join(configuration_folder, f'detailed_metrics.json')
            with open(path_detailed_metrics, 'w') as file:
                json.dump(detailed_metrics, file, default=self.convert_ndarray_to_list, indent=4)

            statistics = self.calculate_metrics_statistics(detailed_metrics)

            # Almacenar métricas estadísticas para esta configuración
            models_metrics[model_name] = statistics

            # Imprimir los resultados
            if self.verbose >= 3:
                print(f"Valores medios de las métricas para la configuración {model_name}: ")
                for metric_name, values in statistics.items():  
                    print(f"Métrica: {metric_name}")
                    print(f"\tMedia: {values['mean']}")
                    print(f"\tDesviación Estándar: {values['std_dev']}")
                    print()

            conf_plot = self.plot_confusion_matrix(models_metrics[model_name]['confusion_matrix']['mean'], models_metrics[model_name]['confusion_matrix']['std_dev'], self.classes, NUM_SIMULATIONS)
            pio.write_image(conf_plot, os.path.join(configuration_folder, f'confusion_matrix_{model_name}_plot.png'))

            end_time = time.time()  # Tiempo de fin para esta configuración
            elapsed_time = end_time - start_time  # Calcular tiempo transcurrido
            models_metrics[model_name]['execution_time'] = elapsed_time  # Almacenar tiempo transcurrido para esta configuración
            if self.verbose >= 2:
                print(f"Tiempo de ejecución para configuración '{model_name}': {elapsed_time} segundos")

        return models_metrics


    # Función para calcular la media y la desviación estándar de las métricas
    def calculate_metrics_statistics(self, metrics_per_config):
        metrics_values = {}

        for sim, folds in metrics_per_config.items():
            for fold, metrics in folds.items():
                for metric_name, metric_values in metrics['metrics'].items():
                    if metric_name not in metrics_values:
                        metrics_values[metric_name] = []

                    metrics_values[metric_name].extend(metric_values)

        # Calcular la media y la desviación estándar para cada métrica
        metrics_mean_std = {}
        for metric_name, values in metrics_values.items():
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            metrics_mean_std[metric_name] = {'mean': mean, 'std_dev': std}

        return metrics_mean_std

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

    def plot_confusion_matrix(self, mean_confusion_matrix, std_dev_confusion_matrix, classes, NUM_SIMULATIONS):
        """
        Plots the confusion matrix.

        Args:
            mean_confusion_matrix (numpy.ndarray): Mean confusion matrix values.
            std_dev_confusion_matrix (numpy.ndarray): Standard deviation of confusion matrix values.
            classes (list): List of class labels.

        Returns:
            plotly.graph_objects.Figure: Plotly figure object.
        """

        total_samples = np.sum(mean_confusion_matrix)

        mean_confusion_matrix_percent = (mean_confusion_matrix / total_samples) * 100
        std_dev_confusion_matrix_percent = (std_dev_confusion_matrix / total_samples) * 100

        fig = go.Figure(data=go.Heatmap(
        z=mean_confusion_matrix_percent,
        zmin=0,
        zmax=100,
        hoverongaps=False,
        colorscale='Blues',
        colorbar=dict(title='Percentage')))

        annotations = []

        for i in range(mean_confusion_matrix.shape[0]):
            for j in range(mean_confusion_matrix.shape[1]):
                annotation = {
                    'x': j,
                    'y': i,
                    'text': f"{mean_confusion_matrix_percent[i, j]:.2f}% ± {std_dev_confusion_matrix_percent[i, j]:.2f}%",
                    'showarrow': False,
                    'font': {'color': 'black'}
                }
                annotations.append(annotation)

        fig.update_layout(
            title='Confusion Matrix (Mean +/- Standard Deviation)',
            xaxis=dict(title='Predicted Labels', tickvals=list(range(len(classes))), ticktext=classes),
            yaxis=dict(title='True Labels', tickvals=list(range(len(classes))), ticktext=classes),
            annotations=annotations)
        
        fig.add_annotation(
            text=f"Total samples\nin {NUM_SIMULATIONS} iterations: {total_samples*NUM_SIMULATIONS}",
            xref="paper", yref="paper",
            x=1.2, y=1.2,
            showarrow=False,
            font=dict(size=12),
            align='right')

        return fig
    
    def perform_test_suite(self, df, tags, metadata, modelo_folder, thresholds):
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
        configuration_folder = os.path.join(simulation_date_folder, f"model_{config_name}")
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
    
    def SetUp_CloudWorkSpace(self, path):
        # Cargar configuraciones desde un archivo YAML
        with open(path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        
        evidently_config = yaml_content.get('evidently', {})
        token = evidently_config.get('token', None)
        url = evidently_config.get('url', None)
        project_id = evidently_config.get('project_id', None)

        ws = CloudWorkspace(token=token, url=url)

        return ws, project_id
    
    # Definir una función para construir los modelos
    def build_models(self, configurations, architectures):
        models = []
        for config_key, config in configurations.items():
            for arch_key, arch in architectures.items():
                # Crear una instancia de ConvNetBuilder para cada combinación de configuración y arquitectura
                builder = ConvNetBuilder(
                    kernel_widths=config['kernel_widths'],
                    filters=config['filters'],
                    dropouts=config['dropouts'],
                    layer_types=arch,
                    tags=config['tags'],
                    metadata=config['metadata'],
                    test_suite_thresholds=config['test_suite_thresholds']
                )
                # Construir el modelo y añadirlo a la lista de modelos
                model, tags, metadata, test_suite_thresholds = builder.build_model()
                model_name = f"{config_key}_{arch_key}"
                models.append([model_name, model, tags, metadata, test_suite_thresholds])

        return models

    
    def convert_ndarray_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def load_yaml(self, filename):
        with open(filename, 'r') as file:
            return yaml.safe_load(file)
