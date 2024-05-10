import yaml
import re
import tensorflow as tf

from Builder import ConvNetBuilder

class ModelCreate():
    def __init__(self, configs_path):
        with open(configs_path, 'r') as file:
            yaml_content = yaml.safe_load(file)

        self.configurations = yaml_content['configurations'] 
        self.architectures = yaml_content['architectures']

    def create_models(self):
        models = []

        for arch_key, arch in self.architectures.items():
            # Add neurons to architecture
            arch = self.process_arch(arch)

            for config_key, config in self.configurations.items():
                
                kernel_widths, filters, dropouts = self.process_config(config, arch)

                builder = ConvNetBuilder(
                        kernel_widths=kernel_widths,
                        filters=filters,
                        dropouts=dropouts,
                        layer_types=arch,
                        tags=config['tags'],
                        metadata=config['metadata'],
                        test_suite_thresholds=config['test_suite_thresholds']
                    )
                # Construir el modelo y añadirlo a la lista de modelos
                model, tags, metadata, test_suite_thresholds = builder.build_model()

                # Compilar el modelo
                learning_rate = 2e-4
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                loss = tf.keras.losses.categorical_crossentropy
                model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

                model_name = f"{config_key}_{arch_key}"

                models.append([model_name, model, tags, metadata, test_suite_thresholds])

        return models

    def process_config(self, config, arch):
        NUM_CONVOLUTIONS = arch.count('C')
        NUM_DROPOUTS = arch.count('D')

        if re.match(r'\d{1,2}_cte', config['kernel_widths']):
            kernel_widths = self.process_kernel_width_cte(NUM_CONVOLUTIONS, config)
        elif re.match(r'(ASC|DESC)', config['kernel_widths']):
            kernel_widths = self.process_asc_desc(NUM_CONVOLUTIONS, config)
        else:
            kernel_widths = []
            print("Invalid kernel_widths value")

        if re.match(r'\d{2}_cte', config['filters']):
            filters = self.process_filters(NUM_CONVOLUTIONS, config)
        else:
            filters = []
            print("Invalid filters value")

        if re.match(r'0\.\d{1,2}_cte', config['dropouts']):
            dropouts = self.process_dropouts(NUM_DROPOUTS, config)
        else:
            dropouts = []
            print("Invalid dropouts value")

        return kernel_widths, filters, dropouts
    

    def process_kernel_width_cte(self, NUM_CONVOLUTIONS, config):
        kernel_widths_str = config['kernel_widths']
        kernel_widths_value = int(kernel_widths_str.split('_')[0])
        # Construir el array de kernel_widths
        kernel_widths = [kernel_widths_value] * NUM_CONVOLUTIONS
        return kernel_widths

    def process_filters(self, NUM_CONVOLUTIONS, config):
        filters_str = config['filters']
        filters_value = int(filters_str.split('_')[0])
        # Construir el array de filters
        filters = [filters_value] * NUM_CONVOLUTIONS
        return filters

    def process_dropouts(self, NUM_DROPOUTS, config):
        dropouts_str = config['dropouts']
        dropouts_value = float(dropouts_str.split('_')[0])
        # Construir el array de dropouts
        dropouts = [dropouts_value] * NUM_DROPOUTS
        return dropouts

    def process_asc_desc(self, NUM_CONVOLUTIONS, config):
        growth_direction = config['kernel_widths']
        if growth_direction == 'ASC':
            # Si crece, el primer valor será 5 y el último 20
            paso = (20 - 5) / (NUM_CONVOLUTIONS - 1)
            kernel_widths = [int(5 + i * paso) for i in range(NUM_CONVOLUTIONS)]
        elif growth_direction == 'DESC':
            # Si decrece, el primer valor será 20 y el último 5
            paso = (5 - 20) / (NUM_CONVOLUTIONS - 1)
            kernel_widths = [int(20 + i * paso) for i in range(NUM_CONVOLUTIONS)]
            
        else:
            kernel_widths = []

        return kernel_widths
    
    def process_arch(self, arch):
        NUM_DENSES = arch.count('F')
        step = (2048 - 2) / (NUM_DENSES - 1) if NUM_DENSES > 1 else 2046  # Manejo de casos especiales
        neurons_array = [int(2048 - i * step) for i in range(NUM_DENSES)]
        arch.extend(neurons_array)

        return arch

