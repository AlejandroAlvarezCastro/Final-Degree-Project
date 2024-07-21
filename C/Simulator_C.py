from __future__ import print_function
import keras
import tensorflow as tf
import plotly.io as pio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import KFold
import os
import time
import json

from CreateModel import ModelCreate
from func import *

from evidently.ui.workspace.cloud import CloudWorkspace

class Simulator:
    def __init__(self, yaml_path, X, Y, ass_f, ass_l, verbose=0):
        self.X = X
        self.Y = Y
        self.ass_f = ass_f
        self.ass_l = ass_l
        self.yaml_path = yaml_path
        self.verbose = verbose
        self.classes = ['OK', 'NOK']
        self.batch_size = 16
        self.epochs = 50
        # self.Reference1 = np.load('/home/aacastro/Alejandro/DQ_ACA_2024/C/data/square/NewReference5.npz') # For square images
        self.OK_manhattan = np.load('/home/aacastro/Alejandro/DQ_ACA_2024/C/data/OK_manhattan.npy')
        self.NOK_manhattan = np.load('/home/aacastro/Alejandro/DQ_ACA_2024/C/data/NOK_manhattan.npy')    
        
        # Setting up the cloud working space environment
        self.ws, self.project_id = self.SetUp_CloudWorkSpace(yaml_path)

    def simulate(self, NUM_SIMULATIONS, NUM_FOLDERS, results_base_directory):
        """
        Executes model training and evaluation simulations.

        Args:
            NUM_SIMULATIONS (int): Number of simulations to run.
            NUM_FOLDERS (int): Number of folders for cross-validation
            results_base_directory (str): Base path to store results

        Returns:
            models_metrics (dict): Dictionary with mean results for each model simulated
        """

        Model = ModelCreate(self.yaml_path)
        models = Model.create_models()

        # Inicializar diccionario para almacenar métricas de los modelos
        models_metrics = {}

        input_shape = (264, 18)
        margin = 1

        store_models_temp = {}
        for sim in range(1, NUM_SIMULATIONS + 1):
            store_models_temp[f'sim_{sim}'] = {}
            for fold in range(1, NUM_FOLDERS + 1):
                store_models_temp[f'sim_{sim}'][f'fold_{fold}'] = []
        
        detailed_metrics = generate_empty_dict(NUM_FOLDERS, NUM_SIMULATIONS)

        # Iterar sobre los modelos construidos
        for k in range(len(models)):
            model_name, model, tags, metadata, thresholds = models[k]

            start_time = time.time()

            for i in range (NUM_SIMULATIONS):
                kf = KFold(n_splits=NUM_FOLDERS, shuffle=True)
                for fold_no, (train, test) in enumerate(kf.split(self.X, self.Y), 1):

                    simulation_date_folder = os.path.join(results_base_directory, f"fecha_simulacion_{time.strftime('%Y%m%d')}")
                    os.makedirs(simulation_date_folder, exist_ok=True)
                    configuration_folder = os.path.join(simulation_date_folder, f"{model_name}")
                    os.makedirs(configuration_folder, exist_ok=True)

                    embedding_network = model
                    input_1 = keras.layers.Input(input_shape)
                    input_2 = keras.layers.Input(input_shape) 
                    tower_1 = embedding_network(input_1)
                    tower_2 = embedding_network(input_2)
                    merge_layer = keras.layers.Lambda(euclidean_distance, output_shape=(1,))([tower_1, tower_2])
                    normal_layer = keras.layers.BatchNormalization()(merge_layer)
                    output_layer = keras.layers.Dense(1, activation="sigmoid")(normal_layer)
                    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
                    siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])

                    X_train, X_test, Y_train, Y_test = train_test_split(self.X[train], self.Y[train], test_size=0.2, random_state=42)
                    X_train = X_train.astype('float32')
                    X_test = X_test.astype('float32')
                    ass_f = self.ass_f.astype('float32')
                    X_train /= 255
                    X_test /= 255
                    ass_f /= 255
                    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
                    del X_train, Y_train
                    pairs_train, labels_train = make_pairs(x_train, y_train)
                    pairs_val, labels_val = make_pairs(x_val, y_val)
                    pairs_test, labels_test = make_pairs(X_test, Y_test)
                    x_train_1 = pairs_train[:, 0] 
                    x_train_2 = pairs_train[:, 1]
                    x_val_1 = pairs_val[:, 0]  
                    x_val_2 = pairs_val[:, 1]
                    x_test_1 = pairs_test[:, 0]
                    x_test_2 = pairs_test[:, 1]
                    
                    history = siamese.fit(
                        [x_train_1, x_train_2],
                        labels_train,
                        validation_data=([x_val_1, x_val_2], labels_val),
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                    )

                    predicted_labels = classify_images(siamese, self.X[test], self.OK_manhattan, self.NOK_manhattan);
            
                    store_simulation_data(detailed_metrics, i, fold_no, train, test, self.Y[test], predicted_labels)

                    store_models_temp[f'sim_{i+1}'][f'fold_{fold_no}'].append(siamese)

            (best_sim_key, best_fold_key), best_value, best_predicted_labels = find_best_f1_score(detailed_metrics)
            print(f"The best model was at simulation {best_sim_key} and fold {best_fold_key} with value {best_value}")

            metadata['folder'] = best_fold_key 
            metadata['num_simulation'] = best_sim_key

            model_to_save = store_models_temp[best_sim_key].get(best_fold_key)

            save_path_model = os.path.join(configuration_folder, f'model_{best_sim_key}_{best_fold_key}.keras')

            best_model = model_to_save[0]
            best_model.save(save_path_model)

            predicted_ass = classify_images(best_model, ass_f, self.OK_manhattan , self.NOK_manhattan )

            df_ass = pd.DataFrame({'target':self.ass_l, 'prediction': predicted_ass})

            perform_report(self.ws, self.project_id, configuration_folder, df_ass, tags, metadata)

            perform_test_suite(self.ws, self.project_id, df_ass, tags, metadata, configuration_folder, thresholds)

            statistics = self.calculate_metrics_statistics(detailed_metrics)

            detailed_metrics['best_model'] = [best_sim_key, best_fold_key]

            path_detailed_metrics = os.path.join(configuration_folder, f'detailed_metrics.json')
            with open(path_detailed_metrics, 'w') as file:
                json.dump(detailed_metrics, file, default=self.convert_ndarray_to_list, indent=4)

            # Almacenar métricas estadísticas para esta configuración
            models_metrics[model_name] = statistics

            conf_plot = self.plot_confusion_matrix(models_metrics[model_name]['confusion_matrix']['mean'], models_metrics[model_name]['confusion_matrix']['std_dev'], self.classes, NUM_SIMULATIONS)
            pio.write_image(conf_plot, os.path.join(configuration_folder, f'confusion_matrix_{model_name}_plot.png'))

            end_time = time.time()  # Tiempo de fin para esta configuración
            elapsed_time = end_time - start_time  # Calcular tiempo transcurrido
            models_metrics[model_name]['execution_time'] = elapsed_time  # Almacenar tiempo transcurrido para esta configuración

        return models_metrics




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

    def convert_ndarray_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
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

