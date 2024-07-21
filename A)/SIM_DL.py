import os, os.path
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

# Create an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser(description='Model training and evaluation simulator.')
parser.add_argument('-r', '--repetitions', type=int, required=True, 
                    help='Number of simulations')
parser.add_argument('-s', '--splits', type=int, required=True,
                     help='Number of folders for cross-validation')
parser.add_argument('-a', '--architecture', type=str, required=True,
                     choices=['3convs', '5convs', '7convs', '10convs'],
                     help='Architecture type (3convs, 5convs, 7convs, 10convs)')
parser.add_argument('-v', '--verbose', type=int, default=0,
                     help='''
                     Verbosity level (0: no messages, 1: specific messages, 
                                      2: general messages, 3: detailed messages, 4: all messages)''')
# Parse the command-line arguments
args = parser.parse_args()

NUM_SIMULATIONS = args.simulations
NUM_FOLDERS = args.folders
ARCHITECTURES = args.architecture
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

if ARCHITECTURES == '3convs':
    yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/A)/configs3convs.yaml'
elif ARCHITECTURES == '5convs':
    yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/A)/configs3convs.yaml'
elif ARCHITECTURES == '7convs':
    yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/A)/configs7convs.yaml'
elif ARCHITECTURES == '10convs':
    yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/A)/configs10convs.yaml'

# Define the base path to store reults
results_base_directory = f"/home/aacastro/Alejandro/DQ_ACA_2024/A)/results/final_sim/{ARCHITECTURES}"

simulator = Simulator(yaml_path, X, Y, ass_f, ass_l, verbose=verbose_level)

model_metrics = simulator.simulate(NUM_SIMULATIONS, NUM_FOLDERS, results_base_directory)

# Printing overall metrics
# print("Overall metrics:")
# for config_name, config_metrics in model_metrics.items():
#     print(f"Configuration: {config_name}")
#     for metric, values in config_metrics.items():
#         if metric != 'execution_time':
#             print(f"{metric}:")
#             print(f"\tMean: {values['mean']}")
#             print(f"\tStandard Deviation: {values['std_dev']}")
#         else:
#             print(f"Execution Time: {model_metrics[config_name]['execution_time']} seconds\n\n")
