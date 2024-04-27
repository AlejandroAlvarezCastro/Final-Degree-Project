import sys, os, datetime, pickle, time, json, yaml
import string, pdb, tqdm
import random, cv2, keras, os.path
import pandas as pd
import numpy as np
import tensorflow as tf
#
from Builder import ConvNetBuilder
from Simulator import Simulator
from CustomReport import CustomReport
#
from timeit import default_timer as timer
from datetime import datetime, timedelta, date
from PIL import Image
from matplotlib import pyplot as plt
from IPython.display import Image

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import ReportFilter
from evidently.ui.dashboards import PanelValue
from evidently.options.base import Options
from evidently.options.option import Option
from evidently.options import ColorOptions
from evidently.options.agg_data import RenderOptions
from evidently.options.color_scheme import NIGHTOWL_COLOR_OPTIONS

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score

from evidently.report import Report
from evidently.metrics import *
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import ClassificationQualityMetric
#
from evidently.test_suite import TestSuite
from evidently.test_preset import BinaryClassificationTestPreset
from evidently.tests import *

tf.autograph.set_verbosity(0)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

#################################################################################################################################
#################################################################################################################################
                                            # CLOUD ENVIROMENT #
#################################################################################################################################
#################################################################################################################################

ws = CloudWorkspace(
token="dG9rbgF5sQ3u67hNVo7e7sL4fnJ4XmqN/OShAb/AhvcKVvPVdwBQ+rfox7G0ehgQfqSW3A+XWdfKYIoQMxrE7TcN0gBIU4Bia8KwfEtqBSx8hWwCrDno1orAqb5vNXp6e8Zr/KXRyydV8/ihwcIxDVePx4dc7XfrXF3i",
url="https://app.evidently.cloud")
project = ws.get_project("773856a9-b047-46c8-8c58-be21716e9369")

#################################################################################################################################
#################################################################################################################################
                                            # READ AND PREPARE DATA #
#################################################################################################################################
#################################################################################################################################

train_data = np.load('/home/aacastro/Alejandro/DQ_ACA_2024/A)/ZN_1D_imgs/orig/train.npz') # CHANGE PATH
train_f = train_data['features']
train_l = train_data['labels']

valid_data = np.load('/home/aacastro/Alejandro/DQ_ACA_2024/A)/ZN_1D_imgs/orig/validation.npz') # CHANGE PATH
valid_f = valid_data['features']
valid_l = valid_data['labels']

test_data = np.load('/home/aacastro/Alejandro/DQ_ACA_2024/A)/ZN_1D_imgs/orig/test.npz') # CHANGE PATH
test_f = test_data['features']
test_l = test_data['labels']

assess_data = np.load('/home/aacastro/Alejandro/DQ_ACA_2024/A)/ZN_1D_imgs/orig/assess.npz') # CHANGE PATH
ass_f = assess_data['features']
ass_l = assess_data['labels']

x_train_valid = np.concatenate((train_f, valid_f), axis=0)
y_train_valid = np.concatenate((train_l, valid_l), axis=0)

X = np.concatenate((x_train_valid, test_f), axis=0)
Y = np.concatenate((y_train_valid, test_l), axis=0)

#################################################################################################################################
#################################################################################################################################
                                            # PREPARE TRAINING #
#################################################################################################################################
#################################################################################################################################

learning_rate   = 2e-4
BATCH_SIZE      = 50
STEPS_PER_EPOCH = train_l.size / BATCH_SIZE
SAVE_PERIOD     = 1
modelPath = os.path.join(os.getcwd(), 'ZN_1D_imgs/bestModel.keras')

loss = tf.keras.losses.categorical_crossentropy

checkpoint = ModelCheckpoint( # set model saving checkpoints
                            modelPath, # set path to save model weights
                            monitor='accuracy', # set monitor metrics
                            verbose=1, # set training verbosity
                            save_best_only=True, # set if want to save only best weights
                            save_weights_only=False, # set if you want to save only model weights
                            mode='auto', # set if save min or max in metrics
                            save_freq= int(SAVE_PERIOD * STEPS_PER_EPOCH) # interval between checkpoints
                            )

earlystopping = EarlyStopping(
        monitor='accuracy', # set monitor metrics
        min_delta=0.0001, # set minimum metrics delta
        patience=25, # number of epochs to stop training
        restore_best_weights=True, # set if use best weights or last weights
        )

#################################################################################################################################
#################################################################################################################################
                                            # START SIMULATION #
#################################################################################################################################
#################################################################################################################################

# Setting up number of simulations and number of folders for cross-validation
NUM_SIMULATIONS = 2
NUM_FOLDERS = 5

# Loading configurations from a YAML file
with open('/home/aacastro/Alejandro/DQ_ACA_2024/A)/configs.yaml') as file:  # CHANGE PATH
    configurations = yaml.safe_load(file)

# Extracting configuration items
config_items = configurations['configurations'].items()

# Dictionary to store metrics for each model
models_metrics = {}

# Iterating through configuration groups
for k in range(0, len(config_items), NUM_FOLDERS):
    # Getting the current group of 5 configurations
    config_group = list(config_items)[k:k+NUM_FOLDERS]
    print(f"Reading group of 5 configurations from {k+1} to {k+5}:")

    # Dictionary to store average metrics for each model
    average_metrics = {"f1_score": [], "recall": [], "precision": [], "roc_auc": [], "confusion_matrix": [] }
    
    # Iterating through simulations
    for i in range(NUM_SIMULATIONS):

        # Setting up K-Fold cross-validation
        kf = KFold(n_splits=NUM_FOLDERS, shuffle=True)

        # Dictionary to store metrics for each fold in the simulation
        metrics_per_simulation = {"f1_score": [], "recall": [], "precision": [], "roc_auc": [], "confusion_matrix": [] }

        # Iterating through folds
        for fold_no, (train, test) in enumerate(kf.split(X, Y), 1):
            # Retrieving configuration parameters for the current fold
            config_name, config_params = config_group[fold_no - 1]
            
            print(f"Name: {config_name}, Parameters: {config_params}, in fold {fold_no}")

            # Building the model using configuration parameters
            builder = ConvNetBuilder(**config_params)
            model, name, tags, metadata = builder.build_model()

            # Compiling the model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

            # Training the model
            print('------------------------------------------------------------------------')
            print(f'Training model {name} in fold {fold_no}...')

            callbacksList = [checkpoint, earlystopping]
            hist = model.fit(X[train], Y[train], epochs=metadata['epochs'], batch_size=metadata['batch_size'], callbacks=callbacksList)

            # Evaluating the model
            ypredt = model.predict(X[test])
            y_pred_labels = np.argmax(ypredt, axis=1)
            pred_prob = ypredt[:, 1]
            yTestClassT = np.argmax(Y[test], axis=1)
            df_cur = pd.DataFrame({'target': yTestClassT, 'prediction': pred_prob})

            # Running a custom classification performance report
            classification_performance_report = CustomReport(metrics=[ClassificationPreset()], tags=tags, metadata=metadata)
            classification_performance_report.run(reference_data=None, current_data=df_cur)
            ws.add_report(project.id, classification_performance_report)

            # Saving the report in JSON format
            nombre_archivo = f"datos_model_{name}_iteration_{i}.json"
            directorio = "/home/aacastro/Alejandro/DQ_ACA_2024/A)/jsons/"
            ruta_completa = os.path.join(directorio, nombre_archivo)
            json_data = classification_performance_report.json()
            classification_performance_report.save(ruta_completa)

            # Computing additional metrics
            f1 = f1_score(yTestClassT, y_pred_labels)
            recall = recall_score(yTestClassT, y_pred_labels)
            precision = precision_score(yTestClassT, y_pred_labels)
            roc_auc = roc_auc_score(yTestClassT, pred_prob)
            conf_mat = confusion_matrix(yTestClassT, y_pred_labels)

            # Printing metrics for this model on this iteration
            print(f"Simulation {i+1}, Folder {fold_no}:")
            print("F1 Score:", f1_score(yTestClassT, y_pred_labels))
            print("Recall:", recall_score(yTestClassT, y_pred_labels))
            print("Precision:", precision_score(yTestClassT, y_pred_labels))
            print("ROC AUC:", roc_auc_score(yTestClassT, pred_prob))
            print("Confusion Matrix:\n", confusion_matrix(yTestClassT, y_pred_labels))

            # Predict on the independent set.
            ypredt_ass = model.predict(ass_f)
            pred_prob_ass = ypredt_ass[:, 1]
            yTestClassT_ass = np.argmax(ass_l, axis=1)
            df_cur_ass = pd.DataFrame({'target':yTestClassT_ass, 'prediction': pred_prob_ass})

            # Performing Test Suite evaluation with different thresholds
            thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            for threshold in thresholds:
                binary_classification_performance = TestSuite(tests=[TestPrecisionScore(gte=threshold), 
                                                                        TestRecallScore(gte=threshold), 
                                                                        TestF1Score(gte=threshold), 
                                                                        TestRocAuc(gte=threshold)], 
                                                                        tags=tags, 
                                                                        metadata=metadata)

                binary_classification_performance.run(current_data=df_cur_ass, reference_data=None)
                ws.add_test_suite(project_id='773856a9-b047-46c8-8c58-be21716e9369', test_suite=binary_classification_performance)

                # Saving the test suite results in json format
                with open(f'test_suite_{name}_iteration_{i}_{threshold}.json', 'w') as file:
                    json.dump(binary_classification_performance.as_dict(), file, indent=4)

            # Storing metrics for this model in this simulation
            if name not in models_metrics:
                models_metrics[name] = {"f1_score": [], "recall": [], "precision": [], "roc_auc": [], "confusion_matrix":[]}

            models_metrics[name]["f1_score"].append(f1)
            models_metrics[name]["recall"].append(recall)
            models_metrics[name]["precision"].append(precision)
            models_metrics[name]["roc_auc"].append(roc_auc)
            models_metrics[name]["confusion_matrix"].append(conf_mat)
        
    # Calculating the mean and standard deviation of metrics for each model
    average_metrics = {}
    for model_name, model_metrics in models_metrics.items():
        average_metrics[model_name] = {}
        for metric, values in model_metrics.items():
            mean_metric = np.mean(values, axis=0)
            std_dev_metric = np.std(values, axis=0)
            average_metrics[model_name][metric] = {"mean": mean_metric, "std_dev": std_dev_metric}

    # Printing average metrics for this group of configurations
    print("Average metrics and standard deviation per model for the current group of configurations:")
    for model_name, metrics in average_metrics.items():
        print(f"Model: {model_name}")
        for metric, values in metrics.items():
            print(f"{metric}:")
            print(f"  Mean: {values['mean']}")
            print(f"  Standard Deviation: {values['std_dev']}")

    print("\n")  # Separator between configuration groups

