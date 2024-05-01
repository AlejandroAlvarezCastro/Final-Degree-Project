import os, os.path
import numpy as np
import tensorflow as tf
import argparse
#
from Simulator import Simulator
#
from evidently.ui.workspace.cloud import CloudWorkspace


tf.autograph.set_verbosity(0)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


# Vervosity value lecture
parser = argparse.ArgumentParser(description='Simulador de entrenamiento y evaluación de modelos.')
parser.add_argument('-v', '--verbose', type=int, default=0, help='Nivel de verbosidad (0: sin mensajes, 1: mensajes generales, 2: mensajes detallados, 3: todos los mensajes)')
args = parser.parse_args()
verbose_level = args.verbose

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
                                            # START SIMULATION #
#################################################################################################################################
#################################################################################################################################

# Setting up number of simulations and number of folders for cross-validation
NUM_SIMULATIONS = 2
NUM_FOLDERS = 2

# Define the base path to store reults
results_base_directory = "/home/aacastro/Alejandro/DQ_ACA_2024/A)/results"

simulator = Simulator(ws, project, X, Y, ass_f, ass_l, verbose=verbose_level)

model_metrics = simulator.simulate(NUM_SIMULATIONS, NUM_FOLDERS, results_base_directory)

# Printing overall metrics
print("Overall metrics:")
for config_name, config_metrics in model_metrics.items():
    print(f"Configuration: {config_name}")
    for metric, values in config_metrics.items():
        if metric != 'execution_time':
            print(f"{metric}:")
            print(f"\tMean: {values['mean']}")
            print(f"\tStandard Deviation: {values['std_dev']}")
        else:
            print(f"Execution Time: {model_metrics[config_name]['execution_time']} seconds")
