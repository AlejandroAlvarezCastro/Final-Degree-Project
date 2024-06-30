import yaml
import re
import tensorflow as tf

from Builder import ConvNetBuilder
from DataProcessor import DataProcessor

class ModelCreate():
    """
    Class to create convolutional neural network models based on configurations and architectures provided in a YAML file.

    Attributes:
        configurations (dict): Dictionary containing configurations for model creation.
        architectures (dict): Dictionary containing architectures for model creation.

    Methods:
        __init__(configs_path): Initializes ModelCreate instance.
        create_models(): Creates models based on configurations and architectures.
        process_config(config, arch): Processes configuration to determine kernel widths, filters, and dropout values.
        process_kernel_width_cte(NUM_CONVOLUTIONS, config): Processes constant kernel widths.
        process_filters(NUM_CONVOLUTIONS, config): Processes filters.
        process_dropouts(NUM_DROPOUTS, config): Processes dropout values.
        process_asc_desc(NUM_CONVOLUTIONS, config): Processes kernel widths in ascending or descending order.
        process_arch(arch): Processes architecture, adding neurones for dense layers.
    """

    def __init__(self, yaml_path):
        """
        Initializes ModelCreate instance.

        Parameters:
            yaml_path (str): Path to the YAML file containing configurations.
        """
        with open(yaml_path, 'r') as file:
            yaml_content = yaml.safe_load(file)

        self.configurations = yaml_content['models'] 
        # self.architectures = yaml_content['architectures']

    def create_models(self):
        """
        Creates models based on configurations and architectures.

        Returns:
            models (list): List of created models.
        """
        models = []
        processor = DataProcessor()

        for config_key, config in self.configurations.items():

            arch = processor.process_arch(config['arch'])
            
            kernel_widths, filters, dropouts, tags, metadata = processor.process_config(config, config_key, arch, config['arch_name'])

            builder = ConvNetBuilder(
                    kernel_widths=kernel_widths,
                    filters=filters,
                    dropouts=dropouts,
                    layer_types=arch,
                    tags=tags,
                    metadata=metadata,
                    test_suite_thresholds=config['test_suite_thresholds']
                )
            # Construir el modelo y añadirlo a la lista de modelos
            model, tags, metadata, test_suite_thresholds = builder.build_model()

            # Compilar el modelo
            # learning_rate = 2e-4
            # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            # loss = tf.keras.losses.binary_crossentropy
            # model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

            model_name = f"{config_key}_{config['arch_name']}"

            models.append([model_name, model, tags, metadata, test_suite_thresholds])

        return models
