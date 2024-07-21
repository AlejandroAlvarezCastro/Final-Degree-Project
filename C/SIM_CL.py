import os, os.path
import tensorflow as tf
import argparse
import numpy as np
#
from C.Simulator_C import Simulator
from Data import Data


# Data for square images
# data = np.load('/home/aacastro/Alejandro/DQ_ACA_2024/C/data/square/train_val_test.npz')
# ass = np.load('/home/aacastro/Alejandro/DQ_ACA_2024/C/data/square/assess.npz')
# X = data['data']
# ass_f = ass['data']
# Y = np.argmax(data['labels'], axis=1)
# ass_l = np.argmax(ass['labels'], axis=1)

# Data for original images
base_path = '/home/aacastro/Alejandro/DQ_ACA_2024/C/ZN_1D_imgs_NO_Aug/'
data_processor = Data(base_path)
X, Y, ass_f, ass_l = data_processor.get_data('train.npz', 'validation.npz', 'test.npz', 'assess.npz')
Y = np.argmax(Y, axis=1)

# Vervosity value lecture
parser = argparse.ArgumentParser(description='Simulador de entrenamiento y evaluación de modelos.')
parser.add_argument('-v', '--verbose', type=int, default=0, help='Nivel de verbosidad ')
parser.add_argument('-r', '--repetitions', type=int, required=True, 
                    help='Number of simulations')
parser.add_argument('-s', '--splits', type=int, required=True,
                     help='Number of folders for cross-validation')
args = parser.parse_args()
verbose_level = args.verbose

NUM_SIMULATIONS = args.simulations
NUM_FOLDERS = args.folders

# Define the base path to store reults
results_base_directory = f"/home/aacastro/Alejandro/DQ_ACA_2024/C/results/square/"
yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/C/models.yaml'

simulator = Simulator(yaml_path, X, Y, ass_f, ass_l, verbose=verbose_level)

model_metrics = simulator.simulate(NUM_SIMULATIONS, NUM_FOLDERS, results_base_directory)
