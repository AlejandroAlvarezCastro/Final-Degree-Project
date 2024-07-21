import sys, os, datetime, pickle, time, yaml, json, re
import string, pdb, tqdm
import random, cv2, keras, os.path
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from collections import Counter

import plotly.graph_objects as go
import plotly.io as pio

from ultralytics import YOLO

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.metric_preset import ClassificationPreset
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.report import Report

import keras

class Simulator:
    """
    Class to simulate model training and evaluate performance metrics.

    Attributes:
        yaml_path (str): Path to the yaml with the information.

        verbose (int): Vervosity for prints.

    Methods:
        simulate(NUM_SIMULATIONS, NUM_FOLDERS, results_base_directory): Simulate configurations training and evaluation.
        calculate_statistics(metrics_list): Calculates mean and standard deviation of metrics from a list.
        plot_metrics(mean_precision, std_dev_precision, mean_recall, std_dev_recall, mean_f1, std_dev_f1, classes): Plots performance metrics.
        plot_confusion_matrix(mean_confusion_matrix, std_dev_confusion_matrix, classes): Plots confusion matrix.
        perform_test_suite(df, tags, metadata, modelo_folder): Performs a Test Suite on data.
        perform_custom_report(model, config_name, df, simulation_date_folder, tags, metadata, fold_no, i): Performs a custom report on data.
        SetUp_CloudWorkSpace(path): Sets up the cloud working space environment
        store_simulation_data(detailed_metrics, i, fold_no, train, test, yTestClassT, y_pred_labels, pred_prob): Stores metrics for fold N and simulation M
        generate_empty_dict(NUM_FOLDERS, NUM_SIMULATIONS): Generates an empty dict to store detailed metrics
    """
    def __init__(self, yaml_path, models_path, results_base_directory, verbose=0):
        self.yaml_path = yaml_path
        self.verbose = verbose
        self.models_path = models_path
        self.results_base_directory = results_base_directory
        self.classes = ['OK', 'NOK']
        self.thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]

        
        # Setting up the cloud working space environment
        self.ws, self.project_id = self.SetUp_CloudWorkSpace(yaml_path)
        

    def simulate(self, NUM_SIMULATIONS, NUM_FOLDERS, CONFIGS):
        """
        Executes model training and evaluation simulations.

        Args:
            NUM_SIMULATIONS (int): Number of simulations to run.
            NUM_FOLDERS (int): Number of folders for cross-validation
            results_base_directory (str): Base path to store results

        Returns:
            models_metrics (dict): Dictionary with mean results for each model simulated
        """

        with open(self.yaml_path, 'r') as file:
            yaml_content = yaml.safe_load(file)

        configurations = yaml_content['configurations'] 

        models_metrics = {}

        for config_key, config in configurations.items():
            model = config['model']
            epochs = int(config['epochs'])
            frozen_layers = int(config['frozen_layers'])
            tags = config['tags']
            metadata = config['metadata']
           
            model = YOLO(f'{self.models_path}/{model}')
            detailed_metrics_test = self.generate_empty_dict(NUM_FOLDERS, NUM_SIMULATIONS)

            start_time = time.time()

            for i in range(NUM_SIMULATIONS):

                # Crear la carpeta de la fecha de la simulación
                simulation_date_folder = os.path.join(self.results_base_directory, f"fecha_simulacion_{time.strftime('%Y%m%d')}")
                os.makedirs(simulation_date_folder, exist_ok=True)

                cross_val_path = f'/home/aacastro/Alejandro/DQ_ACA_2024/B_def/data_cls/full_dataset/{NUM_FOLDERS}-Fold_Cross-val'
                if os.path.exists(cross_val_path):
                    shutil.rmtree(cross_val_path)

                if i+1 == 1 and CONFIGS==1:
                    full_dataset_path = '/home/aacastro/Alejandro/DQ_ACA_2024/B_def/data_cls/full_dataset'
                    self.generate_yolov8_coss_validation(full_dataset_path, NUM_FOLDERS);

                for fold_no in range(1, NUM_FOLDERS + 1):

                    configuration_folder = os.path.join(simulation_date_folder, f"model_{config_key}")
                    os.makedirs(configuration_folder, exist_ok=True)
                    iteration_folder = os.path.join(configuration_folder, f"iteration_{i+1}")
                    os.makedirs(iteration_folder, exist_ok=True)
                    modelo_folder = os.path.join(iteration_folder, f"fold_{fold_no}")
                    os.makedirs(modelo_folder, exist_ok=True)

                    model.train(data=f'/home/aacastro/Alejandro/DQ_ACA_2024/B_def/data_cls/full_dataset/{NUM_FOLDERS}-Fold_Cross-val/split_{fold_no}', 
                                epochs=epochs, freeze=frozen_layers, project=modelo_folder, name=f'train_{fold_no}', patience=20, batch=32)
                    
                    preds_test = {}
                    for clas in self.classes:
                        metrics_pred = model.predict(source=f'/home/aacastro/Alejandro/DQ_ACA_2024/B_def/data_cls/full_dataset/{NUM_FOLDERS}-Fold_Cross-val/split_{fold_no}/test/{clas}', 
                                                model=f'/home/aacastro/Alejandro/DQ_ACA_2024/B_def/results_cls/train_{fold_no}/weights/best.pt')
                        preds_test[clas] = metrics_pred

                    df_test, targets_test, y_pred_labels_test, pred_prob_test = self.get_full_df(preds_test)

                    images_test = self.list_images(f'/home/aacastro/Alejandro/DQ_ACA_2024/B_def/data_cls/full_dataset/{NUM_FOLDERS}-Fold_Cross-val/split_{fold_no}/test')
                    images_train = self.list_images(f'/home/aacastro/Alejandro/DQ_ACA_2024/B_def/data_cls/full_dataset/{NUM_FOLDERS}-Fold_Cross-val/split_{fold_no}/train')

                    self.store_simulation_data(detailed_metrics_test, i, fold_no, images_train, images_test, targets_test, y_pred_labels_test, pred_prob_test, 'test')

            best_sim_key, best_fold_key = self.find_best_f1_score(detailed_metrics_test)
            # if self.verbose >= 3:
            print(f"The best model was at simulation {best_sim_key} and fold {best_fold_key}")

            metadata['folder'] = best_fold_key 
            metadata['num_simulation'] = best_sim_key

            path_detailed_metrics = os.path.join(configuration_folder, f'detailed_metrics_test.json')
            with open(path_detailed_metrics, 'w') as file:
                json.dump(detailed_metrics_test, file, default=self.convert_ndarray_to_list, indent=4)

            statistics = self.calculate_metrics_statistics(detailed_metrics_test)

            models_metrics[config_key] = statistics

            conf_plot = self.plot_confusion_matrix(models_metrics[config_key]['confusion_matrix']['mean'], models_metrics[config_key]['confusion_matrix']['std_dev'], self.classes, NUM_SIMULATIONS)
            pio.write_image(conf_plot, os.path.join(configuration_folder, f'confusion_matrix_{config_key}_plot.png'))

            num_iteration = best_sim_key.split('_')[1]
            num_folder = best_fold_key.split('_')[1]
            model_ass = f'/home/aacastro/Alejandro/DQ_ACA_2024/B_def/results_try/configs_{CONFIGS}/model_{config_key}/iteration_{num_iteration}/fold_{num_folder}/train_{num_folder}/weights/best.pt'
            print("Model_ass: ", model_ass)
                
            metrics_assess = model.predict(source='/home/aacastro/Alejandro/DQ_ACA_2024/B_def/data_cls/assess', 
                                                model=model_ass)
            
            targets_ass = self.obtain_targets(metrics_assess)
            preds_assess = self.obtain_probs(metrics_assess)
            y_pred_labels_ass = self.sim_argmax(preds_assess)
            pred_prob_ass = [pair[1][pair[0].index(0)] for pair in preds_assess]
            df_ass = pd.DataFrame({'target':targets_ass, 'prediction': pred_prob_ass})

            self.perform_report(configuration_folder, df_ass, tags, metadata)

            self.perform_test_suite(df_ass, tags, metadata, configuration_folder, self.thresholds)

            end_time = time.time()  # Tiempo de fin para esta configuración
            elapsed_time = end_time - start_time  # Calcular tiempo transcurrido
            models_metrics[config_key]['execution_time'] = elapsed_time

        path_model_metrics = os.path.join('/home/aacastro/Alejandro/DQ_ACA_2024/B_def', f'model_metrics_{CONFIGS}.json')
        with open(path_model_metrics, 'w') as file:
            json.dump(models_metrics, file, default=self.convert_ndarray_to_list, indent=4)

        return models_metrics

    def generate_empty_dict(self, NUM_FOLDERS, NUM_SIMULATIONS):
        detailed_metrics_test = {}
        for sim in range(1, NUM_SIMULATIONS + 1):
            detailed_metrics_test[f'sim_{sim}'] = {}
            for fold in range(1, NUM_FOLDERS + 1):
                detailed_metrics_test[f'sim_{sim}'][f'fold_{fold}'] = {
                    'metrics': {
                        'f1_score': [],
                        'recall': [],
                        'precision': [],
                        'roc_auc': [],
                        'confusion_matrix': []
                    },
                    'training_indexes': None,
                    'test_indexes': None}
        return detailed_metrics_test 
    

    def list_images(self, directorio_base):
        # Lista para almacenar los nombres de las imágenes sin la extensión
        images_list = []

        # Subdirectorios a revisar
        subdirectorios = ['OK', 'NOK']

        # Recorrer cada subdirectorio
        for subdir in subdirectorios:
            # Construir la ruta completa al subdirectorio
            path = os.path.join(directorio_base, subdir)
            
            # Comprobar si el subdirectorio existe
            if os.path.isdir(path):
                # Listar archivos en el subdirectorio
                for file_name in os.listdir(path):
                    # Comprobar si el archivo es una imagen .png
                    if file_name.endswith('.png'):
                        # Separar el nombre del archivo de su extensión y añadir solo el nombre a la lista
                        nombre_sin_extension = os.path.splitext(file_name)[0]
                        images_list.append(nombre_sin_extension)
        
        return sorted(images_list)
    
    def store_simulation_data(self, detailed_metrics, i, fold_no, train, test, yTestClassT, y_pred_labels, pred_prob, set):
        if set == 'test':
            # Almacenar los índices de las muestras de entrenamiento y prueba en el diccionario
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["training_indexes"] = train
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["test_indexes"] = test

            # Calcular métricas adicionales
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["f1_score"].append(f1_score(yTestClassT, y_pred_labels))
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["recall"].append(recall_score(yTestClassT, y_pred_labels))
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["precision"].append(precision_score(yTestClassT, y_pred_labels))
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["roc_auc"].append(roc_auc_score(yTestClassT, pred_prob))
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["confusion_matrix"].append(confusion_matrix(yTestClassT, y_pred_labels))
        elif set == 'ass':
            # Calcular métricas adicionales
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["f1_score"].append(f1_score(yTestClassT, y_pred_labels))
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["recall"].append(recall_score(yTestClassT, y_pred_labels))
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["precision"].append(precision_score(yTestClassT, y_pred_labels))
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["roc_auc"].append(roc_auc_score(yTestClassT, pred_prob))
            detailed_metrics[f'sim_{i+1}'][f"fold_{fold_no}"]["metrics"]["confusion_matrix"].append(confusion_matrix(yTestClassT, y_pred_labels))


    def sim_argmax(self, preds):
        arg_preds = []
        for k in range(len(preds)):
            arg = int(np.argmax(preds[k]))
            arg_preds.append(arg)
        return arg_preds
    
    def get_full_df(self, preds_test):
        a = preds_test['OK']
        b = preds_test['NOK']

        targets_a = self.obtain_targets(a)
        preds_a = self.obtain_probs(a)
        y_pred_labels_a = self.sim_argmax(preds_a)
        # pred_prob_a =  obtain_prob_class_1(preds_a)
        prob_class_1_a = [pair[1][pair[0].index(0)] for pair in preds_a]
        df_a = pd.DataFrame({'target':targets_a, 'prediction': prob_class_1_a})

        targets_b = self.obtain_targets(b)
        preds_b = self.obtain_probs(b)
        y_pred_labels_b = self.sim_argmax(preds_b)
        # pred_prob_b =  obtain_prob_class_1(preds_b)
        prob_class_1_b = [pair[1][pair[0].index(0)] for pair in preds_b]
        df_b = pd.DataFrame({'target':targets_b, 'prediction': prob_class_1_b})

        df_full =  pd.concat([df_a, df_b])
        targets_full = targets_a + targets_b
        y_pred_labels_full = np.concatenate((y_pred_labels_a, y_pred_labels_b))
        pred_prob_full = np.concatenate((prob_class_1_a, prob_class_1_b))

        return df_full, targets_full, y_pred_labels_full, pred_prob_full

    def obtain_number_from_filename(self, ruta):
        nombre_archivo = os.path.basename(ruta)
        nombre_sin_extension, extension = os.path.splitext(nombre_archivo)
        partes = nombre_sin_extension.split('_')
        numero = partes[-1]
        return numero

    def obtain_targets(self, metrics_pred):
        targets = []
        for k in range(len(metrics_pred)):
            path = metrics_pred[k].path
            target_label = self.obtain_number_from_filename(path)
            targets.append(target_label)
        targets = [int(num) for num in targets]
        return targets

    def obtain_probs(self, metrics_pred):
        reordered_probs_list = []    
        for k in range(len(metrics_pred)):
            label= metrics_pred[k].probs.top5
            array = metrics_pred[k].probs.top5conf.cpu().numpy()
            data_pred = [label, array]
            reordered_probs_list.append(data_pred)
        return reordered_probs_list

    def obtain_prob_class_1(self, probs):
        segundos_elementos = []
        for sublist in probs:
            segundo_elemento = sublist[0][1]
            segundos_elementos.append(segundo_elemento)
        return segundos_elementos
    
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
    
    def perform_report(self, configuration_folder, df, tags, metadata):
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
        classification_performance_report = Report(metrics=[ClassificationPreset()], tags=tags, metadata=metadata)
        classification_performance_report.run(reference_data=None, current_data=df)
        self.ws.add_report(self.project_id, classification_performance_report)

        nombre_archivo_rep = f"custom_report.json"
        ruta_completa = os.path.join(configuration_folder, nombre_archivo_rep)
        classification_performance_report.save(ruta_completa)

    
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
            yaxis=dict(title='True Labels', tickvals=list(range(len(classes))), ticktext=classes, autorange="reversed"),
            annotations=annotations)
        
        fig.add_annotation(
            text=f"Total samples\nin {NUM_SIMULATIONS} iterations: {total_samples*NUM_SIMULATIONS}",
            xref="paper", yref="paper",
            x=1.2, y=1.2,
            showarrow=False,
            font=dict(size=12),
            align='right')

        return fig
    
    def perform_test_suite(self, df, tags, metadata, configuration_folder, thresholds):
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
            ruta_completa_test = os.path.join(configuration_folder, f'test_suite_threshold_{threshold}.json')
            binary_classification_performance.save(ruta_completa_test)       

    def generate_yolov8_coss_validation(self, dataset_path, NUM_FOLDERS):
        # Replace with the path to your dataset
        dataset_path = Path(dataset_path)
        labels = sorted(dataset_path.rglob("*.txt"))  # all data in 'labels'

        cls_idx = [0, 1]

        # Create DataFrame to store label counts
        indx = [l.stem for l in labels]  # uses base filename as ID (no extension)
        labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

        for label in labels:
            lbl_counter = Counter()

            with open(label, 'r') as lf:
                lines = lf.readlines()

            for l in lines:
                # classes for YOLO label uses integer at first position of each line
                lbl_counter[int(l.split(' ')[0])] += 1

            labels_df.loc[label.stem] = lbl_counter

        labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

        ksplit = NUM_FOLDERS
        kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)   # setting random_state for repeatable results

        kfolds = list(kf.split(labels_df))

        folds = [f'split_{n}' for n in range(1, ksplit + 1)]
        folds_df = pd.DataFrame(index=indx, columns=folds)

        for idx, (train, val) in enumerate(kfolds, start=1):
            folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
            folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'test'


        fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

        for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
            train_totals = labels_df.iloc[train_indices].sum()
            val_totals = labels_df.iloc[val_indices].sum()

            # To avoid division by zero, we add a small value (1E-7) to the denominator
            ratio = val_totals / (train_totals + 1E-7)
            fold_lbl_distrb.loc[f'split_{n}'] = ratio

        # Initialize an empty list to store image file paths
        images = []
        images = sorted((dataset_path.rglob("*.png")))

        # Create the necessary directories and dataset YAML files (unchanged)
        save_path = Path(dataset_path / f'{ksplit}-Fold_Cross-val')
        save_path.mkdir(parents=True, exist_ok=True)

        # Copy images to respective split directories
        for n, (train_indices, test_indices) in enumerate(kfolds, start=1):
            split_dir = save_path / f'split_{n}'

            # Create directories for train and test images
            (split_dir / 'train' / 'OK').mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'NOK').mkdir(parents=True, exist_ok=True)
            (split_dir / 'test' / 'OK').mkdir(parents=True, exist_ok=True)
            (split_dir / 'test' / 'NOK').mkdir(parents=True, exist_ok=True)

        for image, label in zip(images, labels):

            s = image.stem
            parts = s.split("_")
            part1 = "_".join(parts[:-1])
            part2 = parts[-1]

            for split, k_split in folds_df.loc[part1].items():
                # Destination directory
                img_to_path_OK = save_path / split / k_split / 'OK'
                img_to_path_NOK = save_path / split / k_split / 'NOK'
                lbl = part2

                if lbl == '0':
                    shutil.copy(image, img_to_path_OK / image.name)
                elif lbl == '1':
                    shutil.copy(image, img_to_path_NOK / image.name)
                else:
                    None

    def convert_ndarray_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def remove_elements(self, lst):
        # Pattern that matches 'sim_{number}' or 'folder_{number}'
        pattern = re.compile(r'^(sim|folder)_\d+$')
        
        # Create a new list excluding elements that match the pattern
        new_list = [element for element in lst if not pattern.match(element)]
        
        return new_list
    
    def find_best_f1_score(self, detailed_metrics):
        """
        Encuentra las claves de simulación y fold que tienen el mayor f1_score.

        Args:
            detailed_metrics (dict): Diccionario con las métricas detalladas.

        Returns:
            tuple: Claves de la simulación y el fold con el mayor f1_score.
        """
        best_f1_score = -1
        best_keys = (None, None)

        for sim_key, sim_value in detailed_metrics.items():
            for fold_key, fold_value in sim_value.items():
                f1_scores = fold_value['metrics']['f1_score']
                if f1_scores:  # Asegurarse de que la lista no esté vacía
                    max_f1_score = max(f1_scores)
                    if max_f1_score > best_f1_score:
                        best_f1_score = max_f1_score
                        best_keys = (sim_key, fold_key)

        return best_keys
