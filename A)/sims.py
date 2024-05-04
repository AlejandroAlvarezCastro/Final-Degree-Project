import os, os.path
import numpy as np
import tensorflow as tf
import argparse
#
from Simulator import Simulator
from Data import Data


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
                                            # READ AND PREPARE DATA #
#################################################################################################################################
#################################################################################################################################

base_path = '/home/aacastro/Alejandro/DQ_ACA_2024/A)/ZN_1D_imgs/orig/'
data_processor = Data(base_path)
X, Y, ass_f, ass_l = data_processor.get_data('train.npz', 'validation.npz', 'test.npz', 'assess.npz')

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
yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/A)/configs.yaml'

simulator = Simulator(yaml_path, X, Y, ass_f, ass_l, verbose=verbose_level)

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
            print(f"Execution Time: {model_metrics[config_name]['execution_time']} seconds\n\n")
