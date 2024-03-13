import tensorflow as tf

class ConvNetBuilder:
    def __init__(self, kernel_widths, filters, dropouts, layer_types):
        self.kernel_widths = kernel_widths
        self.filters = filters
        self.dropouts = dropouts
        self.layer_types = layer_types
        self.integer_indices = []  # List to store the indices of integer elements

        # Check consistency in list sizes
        num_f_layers = layer_types.count('F')
        num_int_values = sum(1 for val in layer_types if isinstance(val, int))
        if num_f_layers != num_int_values:
            raise ValueError("The number of integer values at the end of the layer_types list must be equal to the number of dense layers.")

        # Check that there are only non-zero values in positions where layer_types is 'C'
        for idx, layer_type in enumerate(layer_types):
            if layer_type == 'C':
                if kernel_widths[idx] == 0 or filters[idx] == 0:
                    raise ValueError("In positions where layer_type is 'C', the values in the kernel_widths and filters lists must be non-zero.")

        # Check that kernel_widths and filters lists have at least as many elements as there are elements in layer_types up to the last 'C' included
        last_conv_idx = len(layer_types) - 1 - layer_types[::-1].index('C')
        if len(kernel_widths) < last_conv_idx + 1 or len(filters) < last_conv_idx + 1:
            raise ValueError("The kernel_widths and filters lists must have at least as many elements as there are elements in layer_types up to the last 'C' included.")

    def find_integer_indices(self):
        # Iterate over the layer_types list and obtain the indices of the integer elements
        for i, element in enumerate(self.layer_types):
            if isinstance(element, int):
                self.integer_indices.append(i)

    def build_model(self):

        model = tf.keras.Sequential()

        self.find_integer_indices()  # Calculate the indices of integer elements

        denses = 0  # Counter for dense layers
        drops = 0  # Counter for dropout layers

        for idx, layer_type in enumerate(self.layer_types):
            if layer_type == 'C':  # Convolutional layer
                model.add(tf.keras.layers.Conv1D(filters=self.filters[idx], kernel_size=self.kernel_widths[idx], activation='relu'))
            elif layer_type == 'N':  # Normalization layer
                model.add(tf.keras.layers.BatchNormalization())
            elif layer_type == 'P':  # Pooling layer
                model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
            elif layer_type == 'flatten':  # Flatten layer
                model.add(tf.keras.layers.Flatten())
            elif layer_type == 'D':  # Dropout layer
                if drops < len(self.dropouts):  # Check if dropout rates are available
                    model.add(tf.keras.layers.Dropout(self.dropouts[drops]))
                    drops += 1
                else:
                    print("Not enough dropout rates available to add dropout layers.")
            elif layer_type == 'F':  # Dense layer
                if denses < len(self.integer_indices):  # Check if integer indices are available
                    neurons = self.layer_types[self.integer_indices[denses]]  # Get the number of neurons
                    if denses == len(self.integer_indices) - 1:  # If it's the last dense layer
                        model.add(tf.keras.layers.Dense(neurons, activation='softmax'))  # Use softmax
                    else:
                        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
                    denses += 1
                else:
                    print("Not enough integer indices available to add dense layers.")

            # Rest of the code for other layer types

        return model
