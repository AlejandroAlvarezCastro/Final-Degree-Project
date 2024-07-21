import re

class DataProcessor():
    def __init__(self):
        print('')

    def process_config(self, config, config_key, arch, arch_key):
        """
        Processes configuration to determine kernel widths, filters, and dropout values.

        Parameters:
            config (dict): Configuration for a specific model.
            arch (list): Architecture of the network.

        Returns:
            kernel_widths (list): List of kernel widths.
            filters (list): List of filters.
            dropouts (list): List of dropout values.
        """
        NUM_CONVOLUTIONS = arch.count('C')
        NUM_DROPOUTS = arch.count('D')
        NUM_DENSES = arch.count('F')
        NUM_NORMALIZATION = arch.count('N')
        NUM_POOLING = arch.count('P')

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

        if NUM_NORMALIZATION == 0:
            normalization = 'None'
        elif NUM_NORMALIZATION == 1:
            normalization = 'Parcial'
        else:
            normalization = 'Full'

        if NUM_POOLING == 0:
            pooling = 'None'
        elif NUM_POOLING == 1 and NUM_CONVOLUTIONS != 3:
            pooling = 'Parcial'
        else:
            pooling = 'Full'

        if NUM_DROPOUTS == 0:
            dropout = 'None'
        elif NUM_DROPOUTS == NUM_DENSES:
            dropout = 'Parcial'
        else:
            dropout = 'Full'
    

        current_tags = [f'A#{arch_key}', f'NC#{NUM_CONVOLUTIONS}', f'NF#{NUM_DENSES}', f'N_{normalization}', f'P_{pooling}', f'D_{dropout}',
                        f'KW#{config["kernel_widths"]}', f'F#{config["filters"]}', f'RD#{config["dropouts"]}']
        
        current_metadata = {'arch_name': arch_key, 'arch': arch, 'config': config_key, 'Num_conv': NUM_CONVOLUTIONS, 'Num_denses': NUM_DENSES, 'Num_pooling': NUM_POOLING,
                            'Num_normalization': NUM_NORMALIZATION, 'Num_droputs': NUM_DROPOUTS, 'KW': config["kernel_widths"], 'Filters': config["filters"], 'drop_ratio': config["dropouts"]}


        return kernel_widths, filters, dropouts, current_tags, current_metadata
    

    def process_kernel_width_cte(self, NUM_CONVOLUTIONS, config):
        """
        Processes constant kernel widths.

        Parameters:
            NUM_CONVOLUTIONS (int): Number of convolutional layers in the architecture.
            config (dict): Configuration for a specific model.

        Returns:
            kernel_widths (list): List of kernel widths.
        """
        kernel_widths_str = config['kernel_widths']
        kernel_widths_value = int(kernel_widths_str.split('_')[0])
        # Construir el array de kernel_widths
        kernel_widths = [kernel_widths_value] * NUM_CONVOLUTIONS
        return kernel_widths

    def process_filters(self, NUM_CONVOLUTIONS, config):
        """
        Processes filters.

        Parameters:
            NUM_CONVOLUTIONS (int): Number of convolutional layers in the architecture.
            config (dict): Configuration for a specific model.

        Returns:
            filters (list): List of filters.
        """
        filters_str = config['filters']
        filters_value = int(filters_str.split('_')[0])
        # Construir el array de filters
        filters = [filters_value] * NUM_CONVOLUTIONS
        return filters

    def process_dropouts(self, NUM_DROPOUTS, config):
        """
        Processes dropout values.

        Parameters:
            NUM_DROPOUTS (int): Number of dropout layers in the architecture.
            config (dict): Configuration for a specific model.

        Returns:
            dropouts (list): List of dropout values.
        """
        dropouts_str = config['dropouts']
        dropouts_value = float(dropouts_str.split('_')[0])
        # Construir el array de dropouts
        dropouts = [dropouts_value] * NUM_DROPOUTS
        return dropouts

    def process_asc_desc(self, NUM_CONVOLUTIONS, config):
        """
        Processes kernel widths in ascending or descending order.

        Parameters:
            NUM_CONVOLUTIONS (int): Number of convolutional layers in the architecture.
            config (dict): Configuration for a specific model.

        Returns:
            kernel_widths (list): List of kernel widths.
        """
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
        """
        Processes the architecture of the network, adding neurons for dense layers.

        Parameters:
            arch (list): Architecture of the network.

        Returns:
            arch (list): Modified architecture of the network.
        """
        NUM_DENSES = arch.count('F')
        step = (2048 - 2) / (NUM_DENSES - 1) if NUM_DENSES > 1 else 2046  # Manejo de casos especiales
        neurons_array = [int(2048 - i * step) for i in range(NUM_DENSES)]
        arch.extend(neurons_array)

        return arch