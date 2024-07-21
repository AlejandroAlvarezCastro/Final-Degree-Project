import os, os.path
import tensorflow as tf
import argparse
#
from B_def.Simulator_B import Simulator

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
parser.add_argument('-m', '--model', type=str, required=True,
                     choices=['N', 'S', 'M', 'L'],
                     help='Pretrained model (N, S, M, L)')
parser.add_argument('-v', '--verbose', type=int, default=0,
                     help='''
                     Verbosity level (0: no messages, 1: specific messages, 
                                      2: general messages, 3: detailed messages, 4: all messages)''')
# Parse the command-line arguments
args = parser.parse_args()

NUM_SIMULATIONS = args.simulations
NUM_FOLDERS = args.folders
MODEL = args.architecture
verbose_level = args.verbose


#################################################################################################################################
#################################################################################################################################
                                            # START SIMULATION #
#################################################################################################################################
#################################################################################################################################

if MODEL == 'N':
    yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/B_def/configurationsN.yaml'
elif MODEL == 'S':
    yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/B_def/configurationsS.yaml'
elif MODEL == 'M':
    yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/B_def/configurationsM.yaml'
elif MODEL == 'L':
    yaml_path = '/home/aacastro/Alejandro/DQ_ACA_2024/B_def/configurationsL.yaml'

# Define the base path to store reults
results_base_directory = f"/home/aacastro/Alejandro/DQ_ACA_2024/B_def/results_cls/final_sim/configs_{MODEL}"
models_path = '/home/aacastro/Alejandro/DQ_ACA_2024/B_def/cls-models'

simulator = Simulator(yaml_path, models_path, results_base_directory, verbose_level)

model_metrics = simulator.simulate(NUM_SIMULATIONS, NUM_FOLDERS, MODEL)